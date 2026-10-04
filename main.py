"""图文 PDF RAG 示例。

主流程：
1. 用 PyMuPDF4LLM 提取页面 Markdown，并导出图片文件。
2. 用 PyMuPDF 补充图片页码和 bbox 坐标。
3. 把正文和图片描述统一构建为 Chunk。
4. 用关键词检索召回正文和图片。
5. 有图片命中时，把原图一起发送给视觉模型回答。
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pymupdf
import pymupdf4llm
from dotenv import load_dotenv
from openai import OpenAI


# 默认使用阿里百炼 OpenAI 兼容接口。
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEXT_MODEL = "qwen-plus"
DEFAULT_VISION_MODEL = "qwen-vl-max"


@dataclass
class Chunk:
    """统一表示正文块和图片块。"""

    chunk_id: str
    source: str
    page: int
    text: str
    terms: list[str]
    # kind 为 text 时使用 text 检索；kind 为 image 时再用 image_path 发起多模态问答。
    kind: str = "text"
    image_path: str | None = None
    image_bbox: tuple[float, float, float, float] | None = None
    caption: str = ""


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_terms(text: str) -> list[str]:
    """提取中文二元词组、中文字符、英文单词和数字编码。"""
    text = text.lower()
    terms: list[str] = []

    # 英文单词、型号、错误码、版本号等按完整 token 保留。
    terms.extend(
        re.findall(
            r"[a-z0-9]+(?:[-_/\.][a-z0-9]+)*",
            text,
        )
    )

    # 中文使用二元词组，兼顾中文分词效果和实现复杂度。
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(sequence) <= 2:
            terms.extend(sequence)
        else:
            terms.extend(
                sequence[index:index + 2]
                for index in range(len(sequence) - 1)
            )

    return list(dict.fromkeys(terms))


def split_text(
    text: str,
    max_chars: int = 1200,
    overlap: int = 180,
) -> list[str]:
    """优先按 Markdown 段落分块，超长段落再按字符窗口切分。"""
    text = normalize_text(text)

    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    chunks: list[str] = []
    current = ""

    # 先按 Markdown 空行切段落，尽量保持语义完整。
    for paragraph in paragraphs:
        candidate = (
            f"{current}\n\n{paragraph}".strip()
            if current
            else paragraph
        )

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        # 单个段落仍然过长时，退化为带 overlap 的字符窗口。
        start = 0

        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            piece = paragraph[start:end].strip()

            if piece:
                chunks.append(piece)

            if end >= len(paragraph):
                break

            start = max(0, end - overlap)

        current = ""

    if current:
        chunks.append(current)

    return chunks


def extract_pdf_pages(pdf_path: str, image_dir: str) -> list[dict]:
    """用 PyMuPDF4LLM 提取页面 Markdown，并导出页面中的图片。"""
    image_dir_path = Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)

    # page_chunks=True 保留页码边界；write_images=True 会导出图片并写入 Markdown 占位符。
    pages = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,
        write_images=True,
        image_path=str(image_dir_path),
        image_format="png",
        show_progress=False,
        table_strategy="lines_strict",
    )

    # 匹配 Markdown 图片占位符，例如 ![](demo.pdf.images/demo.pdf-0001-01.png)。
    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    result: list[dict] = []

    with pymupdf.open(pdf_path) as document:
        for index, page_data in enumerate(pages, start=1):
            metadata = page_data.get("metadata", {})
            # 优先使用 PyMuPDF4LLM 返回的真实页码，缺失时再回退到遍历下标。
            page_number = int(
                metadata.get("page")
                or metadata.get("page_number")
                or index
            )
            page_text = page_data.get("text", "")
            image_references = image_pattern.findall(page_text)
            # 图片路径已经单独保存，正文索引中不应继续保留 Markdown 占位符。
            page_text = normalize_text(image_pattern.sub("", page_text))

            # PyMuPDF 提供图片在页面中的展示坐标，后续可用于高亮或裁剪。
            image_infos = document[page_number - 1].get_image_info(
                xrefs=True
            )
            images: list[dict] = []

            for image_index, image_reference in enumerate(
                image_references
            ):
                image_path = (
                    image_dir_path / Path(image_reference).name
                ).resolve()
                image_info = (
                    image_infos[image_index]
                    if image_index < len(image_infos)
                    else {}
                )
                # 某些 PDF 的图片信息不完整，缺失坐标时使用零矩形兜底。
                bbox = image_info.get(
                    "bbox",
                    (0.0, 0.0, 0.0, 0.0),
                )
                images.append(
                    {
                        "page": page_number,
                        "path": str(image_path),
                        "bbox": tuple(
                            float(value) for value in bbox
                        ),
                    }
                )

            result.append(
                {
                    "page": page_number,
                    "text": page_text,
                    "images": images,
                }
            )

    return result


def build_chunks(
    pdf_path: str,
    image_dir: str | None = None,
    image_captioner: Callable[[str, int], str] | None = None,
) -> list[Chunk]:
    """把页面正文和图片描述转换成统一的检索块。"""
    image_dir = image_dir or f"{pdf_path}.images"
    pages = extract_pdf_pages(pdf_path, image_dir=image_dir)
    chunks: list[Chunk] = []

    for page_data in pages:
        page_number = page_data["page"]

        # 正文块只负责文本检索。
        for chunk_index, text in enumerate(
            split_text(page_data["text"])
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"page-{page_number}-chunk-{chunk_index}",
                    source=str(pdf_path),
                    page=page_number,
                    text=text,
                    terms=extract_terms(text),
                    kind="text",
                )
            )

        # 图片块同时承载“可检索描述”和“原始图片路径”。
        for image_index, image in enumerate(page_data["images"]):
            image_path = image["path"]
            if image_captioner:
                try:
                    # 视觉描述只是召回辅助信息，失败不能中断整个 PDF 的索引。
                    caption = image_captioner(
                        image_path,
                        page_number,
                    )
                except Exception as exc:
                    print(
                        f"第 {page_number} 页图片描述失败，"
                        f"使用基础索引：{exc}"
                    )
                    caption = f"第 {page_number} 页图片"
            else:
                # 离线模式或未配置视觉模型时，保留基础图片索引。
                caption = f"第 {page_number} 页图片"

            caption = normalize_text(caption) or (
                f"第 {page_number} 页图片"
            )
            # 把图片描述、页码和文件名放在同一个 chunk 中，便于关键词检索。
            searchable_text = normalize_text(
                f"{caption}\n"
                f"图片来源：第 {page_number} 页 "
                f"{Path(image_path).name}"
            )
            chunks.append(
                Chunk(
                    chunk_id=f"page-{page_number}-image-{image_index}",
                    source=str(pdf_path),
                    page=page_number,
                    text=searchable_text,
                    terms=extract_terms(searchable_text),
                    kind="image",
                    image_path=image_path,
                    image_bbox=image["bbox"],
                    caption=caption,
                )
            )

    return chunks


def save_chunks(chunks: list[Chunk], output_path: str) -> None:
    """把 Chunk 列表序列化为 UTF-8 JSON 缓存。"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(
            [asdict(chunk) for chunk in chunks],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_chunks(input_path: str) -> list[Chunk]:
    """从 JSON 缓存恢复 Chunk，并兼容列表形式的 bbox。"""
    data = json.loads(
        Path(input_path).read_text(encoding="utf-8")
    )
    chunks: list[Chunk] = []

    for item in data:
        if item.get("image_bbox"):
            item["image_bbox"] = tuple(item["image_bbox"])
        chunks.append(Chunk(**item))

    return chunks


def is_image_question(query: str) -> bool:
    """粗略判断用户问题是否存在图片意图，用于给图片块加权。"""
    normalized = query.lower()
    markers = (
        "图片",
        "图像",
        "截图",
        "图表",
        "照片",
        "图中",
        "图里",
        "看图",
        "配图",
        "原图",
        "image",
        "figure",
        "chart",
        "diagram",
    )

    if any(marker in normalized for marker in markers):
        return True

    return re.search(r"(这|那|该|本).{0,4}图", normalized) is not None


class KeywordRetriever:
    """基于中英文词项、IDF 和图片意图加权的轻量检索器。"""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # 记录词项出现在多少个 chunk 中，供 IDF 计算使用。
        self.document_frequency: dict[str, int] = {}

        for chunk in chunks:
            for term in set(chunk.terms):
                self.document_frequency[term] = (
                    self.document_frequency.get(term, 0) + 1
                )

    def _idf(self, term: str) -> float:
        """越少见的词项权重越高，避免“的、是、页”等词影响排序。"""
        document_count = len(self.chunks)
        frequency = self.document_frequency.get(term, 0)
        return math.log(
            (document_count + 1) / (frequency + 1)
        ) + 1.0

    def _score(
        self,
        query: str,
        query_terms: list[str],
        chunk: Chunk,
    ) -> float:
        query_term_set = set(query_terms)

        if not query_term_set:
            return 0.0

        matched_terms = query_term_set & set(chunk.terms)
        # coverage 表示查询词在 chunk 中被覆盖的比例。
        weighted_overlap = sum(
            self._idf(term) for term in matched_terms
        )
        total_weight = sum(
            self._idf(term) for term in query_term_set
        )
        coverage = (
            weighted_overlap / total_weight
            if total_weight > 0
            else 0.0
        )

        normalized_query = normalize_text(query).lower()
        normalized_text = chunk.text.lower()
        exact_bonus = 0.0

        # 完整查询串直接命中时，通常比零散词项命中更相关。
        if normalized_query and normalized_query in normalized_text:
            exact_bonus += 3.0

        # 型号、错误码、版本号等英文数字 token 单独加权。
        for code in set(
            re.findall(
                r"[a-z0-9]+(?:[-_/\.][a-z0-9]+)*",
                normalized_query,
            )
        ):
            if code in normalized_text:
                exact_bonus += 2.0

        # 长文本偶然命中更多词，做轻微长度惩罚；图片问题则额外给图片块加权。
        length_penalty = min(len(chunk.text) / 1000, 2.0)
        image_intent_bonus = (
            4.0
            if chunk.kind == "image"
            and is_image_question(query)
            else 0.0
        )

        return (
            coverage * 10.0
            + exact_bonus
            + length_penalty * 0.1
            + image_intent_bonus
        )

    def search(
        self,
        query: str,
        top_k: int = 6,
        min_score: float = 0.0,
    ) -> list[tuple[float, Chunk]]:
        query_terms = extract_terms(query)
        scored: list[tuple[float, Chunk]] = []

        for chunk in self.chunks:
            score = self._score(
                query=query,
                query_terms=query_terms,
                chunk=chunk,
            )
            if score >= min_score:
                scored.append((score, chunk))

        # 分数相同时保持原始插入顺序，便于文本块和图片块稳定输出。
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]


def format_context(
    results: list[tuple[float, Chunk]],
) -> str:
    if not results:
        return "没有检索到相关资料。"

    sections: list[str] = []

    for index, (score, chunk) in enumerate(results, start=1):
        if chunk.kind == "image":
            image_path = chunk.image_path or ""
            sections.append(
                f"[资料 {index}]\n"
                f"来源文件：{chunk.source}\n"
                f"页码：第 {chunk.page} 页\n"
                f"资料类型：图片\n"
                f"图片路径：{image_path}\n"
                f"检索分数：{score:.4f}\n\n"
                f"{chunk.caption or chunk.text}\n"
            )
        else:
            sections.append(
                f"[资料 {index}]\n"
                f"来源文件：{chunk.source}\n"
                f"页码：第 {chunk.page} 页\n"
                f"资料类型：正文\n"
                f"检索分数：{score:.4f}\n\n"
                f"{chunk.text}\n"
            )

    return "\n\n".join(sections)


def image_to_data_url(image_path: str) -> str:
    """把本地图片编码为 OpenAI Chat Completions 可接收的 Data URL。"""
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def describe_image(
    client: OpenAI,
    model: str,
    image_path: str,
    page: int,
) -> str:
    """调用视觉模型生成图片描述，用于建立图片关键词索引。"""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 PDF 图片索引助手。请用中文简洁描述图片，"
                    "逐字提取图中可见的标题、按钮和关键文字，再说明"
                    "图片类型与主要对象。不要猜测无法确认的信息。"
                ),
            },
            {
                "role": "user",
                # 多模态消息由文本提示和 image_url 图片项组成。
                "content": [
                    {
                        "type": "text",
                        "text": f"请描述 PDF 第 {page} 页的这张图片。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path),
                        },
                    },
                ],
            },
        ],
    )

    return (
        response.choices[0].message.content
        or f"第 {page} 页图片"
    ).strip()


def build_vision_user_content(
    question: str,
    results: list[tuple[float, Chunk]],
) -> list[dict]:
    """构造包含文本资料和原始图片的多模态用户消息。"""
    context = format_context(results)
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "参考资料：\n\n"
                f"{context}\n\n"
                f"用户问题：\n{question}\n\n"
                "请基于参考资料和图片回答，并给出来源页码。"
            ),
        }
    ]
    # 同一张图片可能被多个检索结果引用，上传前去重。
    seen_images: set[str] = set()

    for _score, chunk in results:
        if chunk.kind != "image" or not chunk.image_path:
            continue

        image_path = Path(chunk.image_path)
        if not image_path.is_file():
            continue

        resolved_path = str(image_path.resolve())
        if resolved_path in seen_images:
            continue

        seen_images.add(resolved_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(resolved_path),
                },
            }
        )

    return content


def create_client() -> tuple[OpenAI, str, str]:
    """从 .env 读取百炼配置，并返回文本模型和视觉模型配置。"""
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
    text_model = os.getenv("DASHSCOPE_MODEL", DEFAULT_TEXT_MODEL)
    vision_model = os.getenv(
        "DASHSCOPE_VL_MODEL",
        DEFAULT_VISION_MODEL,
    )

    if not api_key:
        raise RuntimeError(
            "缺少 DASHSCOPE_API_KEY，请检查 .env 文件。"
        )

    return (
        OpenAI(api_key=api_key, base_url=base_url),
        text_model,
        vision_model,
    )


def message_text(message: object) -> str:
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)

    return str(content or "")


def ask_qwen(
    question: str,
    results: list[tuple[float, Chunk]],
) -> str:
    """优先使用视觉模型回答图片问题，失败时降级为纯文本回答。"""
    client, text_model, vision_model = create_client()
    # 约束模型只依据检索资料回答，并强制输出页码引用。
    system_prompt = (
        "你是一个严谨的中文 PDF 文档问答助手。\n\n"
        "回答要求：\n"
        "1. 只能依据“参考资料”回答。\n"
        "2. 如果资料不足，必须回答“资料中没有找到足够信息”。\n"
        "3. 不允许编造资料中不存在的数字、日期、步骤或结论。\n"
        "4. 回答尽量直接，优先使用条目化表达。\n"
        "5. 每个关键结论后标注来源页码，例如：[第 3 页]。\n"
        "6. 如果图片包含答案，要明确说明是从图片中识别到的。"
    )
    context = format_context(results)
    # 只有本地图片文件仍然存在时，才向视觉模型上传原图。
    has_image = any(
        chunk.kind == "image"
        and chunk.image_path
        and Path(chunk.image_path).is_file()
        for _score, chunk in results
    )

    if has_image:
        user_content = build_vision_user_content(question, results)

        try:
            response = client.chat.completions.create(
                model=vision_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return message_text(response.choices[0].message)
        except Exception as exc:
            # 视觉模型不可用时，继续使用已经生成的图片描述回答。
            print(
                f"视觉模型调用失败，将仅使用图片描述回答：{exc}"
            )

    # 无图片或视觉模型失败时，退回到纯文本模型。
    response = client.chat.completions.create(
        model=text_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "参考资料：\n\n"
                    f"{context}\n\n"
                    f"用户问题：\n{question}\n\n"
                    "请基于参考资料回答，并给出来源页码。"
                ),
            },
        ],
    )
    return message_text(response.choices[0].message)


def print_retrieval_results(
    results: list[tuple[float, Chunk]],
) -> None:
    print("\n检索结果：")

    if not results:
        print("没有找到相关内容。")
        return

    for index, (score, chunk) in enumerate(results, start=1):
        preview = chunk.text.replace("\n", " ")[:100]
        chunk_type = "图片" if chunk.kind == "image" else "正文"
        print(
            f"{index}. 第 {chunk.page} 页 | {chunk_type} | "
            f"score={score:.4f} | {preview}..."
        )


def cache_is_usable(
    pdf_path: str,
    cache_path: str,
    image_dir: str,
) -> bool:
    """检查缓存是否比 PDF 新，并且缓存引用的图片文件仍然存在。"""
    pdf_file = Path(pdf_path)
    cache_file = Path(cache_path)

    if not cache_file.exists():
        return False

    if cache_file.stat().st_mtime < pdf_file.stat().st_mtime:
        return False

    try:
        chunks = load_chunks(cache_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False

    image_chunks = [
        chunk for chunk in chunks if chunk.kind == "image"
    ]
    # 图片目录被删除或移动后，旧缓存不能继续使用。
    if image_chunks and not Path(image_dir).is_dir():
        return False

    return all(
        chunk.image_path
        and Path(chunk.image_path).is_file()
        for chunk in image_chunks
    )


def build_or_load_index(
    pdf_path: str,
    cache_path: str,
    image_dir: str | None = None,
    caption_images: bool = True,
    image_captioner: Callable[[str, int], str] | None = None,
) -> list[Chunk]:
    """优先读取可用缓存，否则重新解析 PDF 并建立图文索引。"""
    pdf_file = Path(pdf_path)
    cache_file = Path(cache_path)
    image_dir = image_dir or f"{pdf_path}.images"

    if cache_is_usable(
        pdf_path=pdf_path,
        cache_path=cache_path,
        image_dir=image_dir,
    ):
        print(f"读取缓存：{cache_path}")
        return load_chunks(cache_path)

    print(f"正在解析 PDF：{pdf_path}")
    with pymupdf.open(pdf_path) as document:
        print(f"PDF 页数：{document.page_count}")

    resolved_captioner = image_captioner

    if caption_images and resolved_captioner is None:
        try:
            # 默认在索引阶段调用视觉模型生成图片描述。
            client, _text_model, vision_model = create_client()

            def resolved_captioner(
                path: str,
                page: int,
            ) -> str:
                return describe_image(
                    client=client,
                    model=vision_model,
                    image_path=path,
                    page=page,
                )

        except RuntimeError as exc:
            print(f"未生成图片描述，将使用基础图片索引：{exc}")

    # 记录图片描述失败次数，避免把不完整索引写入长期缓存。
    caption_failures = [0]

    if caption_images and resolved_captioner is not None:
        base_captioner = resolved_captioner

        def tracked_captioner(path: str, page: int) -> str:
            try:
                return base_captioner(path, page)
            except Exception:
                caption_failures[0] += 1
                raise

        resolved_captioner = tracked_captioner

    chunks = build_chunks(
        str(pdf_file),
        image_dir=image_dir,
        image_captioner=(
            resolved_captioner if caption_images else None
        ),
    )
    text_count = sum(chunk.kind == "text" for chunk in chunks)
    image_count = sum(chunk.kind == "image" for chunk in chunks)
    captioning_unavailable = (
        caption_images and resolved_captioner is None
    )
    # 显式关闭描述，或图片描述全部成功时，才允许写缓存。
    should_cache = (
        not caption_images
        or image_count == 0
        or (
            not captioning_unavailable
            and caption_failures[0] == 0
        )
    )

    if should_cache:
        save_chunks(chunks, str(cache_file))

    print(
        f"生成检索块：正文 {text_count}，图片 {image_count}，"
        f"合计 {len(chunks)}"
    )
    if should_cache:
        print(f"写入缓存：{cache_path}")
    else:
        print("图片描述未全部成功，本次不写入缓存。")
    print(f"图片目录：{image_dir}")
    return chunks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 PyMuPDF4LLM、PyMuPDF 和 Qwen 实现图文 PDF RAG"
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="demo.pdf",
        help="PDF 文件路径，默认 demo.pdf",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="索引缓存路径，默认 <pdf>.chunks.json",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="图片导出目录，默认 <pdf>.images",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="只建立索引，不进入问答",
    )
    parser.add_argument(
        "--no-image-captions",
        action="store_true",
        help="不调用视觉模型生成图片描述",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="每次检索的块数量，默认 6",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI 入口：建立索引并进入交互式问答。"""
    args = parse_args(argv)
    pdf_path = args.pdf_path

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"文件不存在：{pdf_path}")

    cache_path = args.cache or f"{pdf_path}.chunks.json"
    image_dir = args.images_dir or f"{pdf_path}.images"
    chunks = build_or_load_index(
        pdf_path=pdf_path,
        cache_path=cache_path,
        image_dir=image_dir,
        caption_images=not args.no_image_captions,
    )

    if not chunks:
        raise RuntimeError(
            "没有提取到文本或图片。"
            "如果这是扫描版 PDF，请先增加 OCR 处理。"
        )

    if args.index_only:
        print("索引完成。")
        return

    retriever = KeywordRetriever(chunks)
    print("\nRAG 已启动，输入 exit 退出。")

    while True:
        question = input("\n问题：").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        if not question:
            continue

        results = retriever.search(
            query=question,
            top_k=args.top_k,
        )
        print_retrieval_results(results)
        print("\n正在调用阿里百炼模型...")
        answer = ask_qwen(
            question=question,
            results=results,
        )
        print("\n回答：")
        print(answer)


if __name__ == "__main__":
    main()
