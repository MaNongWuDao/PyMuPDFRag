from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from main import (
    Chunk,
    KeywordRetriever,
    ask_qwen,
    build_chunks,
    build_or_load_index,
    build_vision_user_content,
    describe_image,
    extract_pdf_pages,
)


class PdfExtractionTests(unittest.TestCase):
    def test_extract_pdf_pages_keeps_text_and_saves_images(self) -> None:
        pdf_path = Path("demo.pdf")

        with tempfile.TemporaryDirectory() as temp_dir:
            pages = extract_pdf_pages(
                str(pdf_path),
                image_dir=temp_dir,
            )

            self.assertEqual(len(pages), 1)

            page = pages[0]
            self.assertEqual(page["page"], 1)
            self.assertIn("Versatile", page["text"])
            self.assertNotIn("![](", page["text"])

            self.assertEqual(len(page["images"]), 1)
            image = page["images"][0]
            self.assertEqual(image["page"], 1)
            self.assertTrue(Path(image["path"]).is_file())
            self.assertEqual(len(image["bbox"]), 4)

    def test_build_chunks_creates_searchable_image_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = build_chunks(
                "demo.pdf",
                image_dir=temp_dir,
                image_captioner=lambda _path, _page: (
                    "The Progressive JavaScript Framework. "
                    "Vue 首页图片，包含 Why Vue、Get Started 和 Install 按钮。"
                ),
            )

            image_chunks = [
                chunk for chunk in chunks if chunk.kind == "image"
            ]
            self.assertEqual(len(image_chunks), 1)

            image_chunk = image_chunks[0]
            self.assertEqual(image_chunk.page, 1)
            self.assertEqual(image_chunk.caption, (
                "The Progressive JavaScript Framework. "
                "Vue 首页图片，包含 Why Vue、Get Started 和 Install 按钮。"
            ))
            self.assertTrue(Path(image_chunk.image_path or "").is_file())

            results = KeywordRetriever(chunks).search(
                "The Progressive JavaScript Framework",
                top_k=1,
            )

            self.assertTrue(results)
            self.assertEqual(results[0][1].kind, "image")

    def test_build_vision_user_content_includes_image_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "demo.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nimage")

            image_chunk = Chunk(
                chunk_id="page-1-image-0",
                source="demo.pdf",
                page=1,
                text="Vue 首页图片",
                terms=["vue", "图片"],
                kind="image",
                image_path=str(image_path),
                image_bbox=(0.0, 0.0, 10.0, 10.0),
                caption="Vue 首页图片",
            )

            content = build_vision_user_content(
                "图片中的标题是什么？",
                [(9.0, image_chunk)],
            )

            self.assertEqual(content[0]["type"], "text")
            self.assertIn("Vue 首页图片", content[0]["text"])
            self.assertEqual(content[1]["type"], "image_url")
            self.assertTrue(
                content[1]["image_url"]["url"].startswith(
                    "data:image/png;base64,"
                )
            )

    def test_search_prioritizes_image_chunk_for_image_question(self) -> None:
        text_chunk = Chunk(
            chunk_id="page-1-chunk-0",
            source="demo.pdf",
            page=1,
            text="文档架构图说明",
            terms=["文档", "档架", "架构", "构图", "图说", "说明"],
        )
        image_chunk = Chunk(
            chunk_id="page-1-image-0",
            source="demo.pdf",
            page=1,
            text="第 1 页图片",
            terms=["第", "页图", "图片"],
            kind="image",
            image_path="demo.png",
            caption="第 1 页图片",
        )

        results = KeywordRetriever(
            [text_chunk, image_chunk]
        ).search("这张架构图是什么？", top_k=1)

        self.assertTrue(results)
        self.assertEqual(results[0][1].kind, "image")

    def test_build_or_load_index_persists_image_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_path = temp_path / "demo.pdf.chunks.json"
            image_dir = temp_path / "images"

            with contextlib.redirect_stdout(io.StringIO()):
                created = build_or_load_index(
                    "demo.pdf",
                    cache_path=str(cache_path),
                    image_dir=str(image_dir),
                    caption_images=False,
                )
                loaded = build_or_load_index(
                    "demo.pdf",
                    cache_path=str(cache_path),
                    image_dir=str(image_dir),
                    caption_images=False,
                )

            self.assertTrue(cache_path.is_file())
            self.assertTrue(any(chunk.kind == "image" for chunk in created))
            self.assertEqual(
                [chunk.chunk_id for chunk in loaded],
                [chunk.chunk_id for chunk in created],
            )
            self.assertTrue(
                Path(
                    next(
                        chunk.image_path
                        for chunk in loaded
                        if chunk.kind == "image"
                    )
                    or ""
                ).is_file()
            )

    def test_build_or_load_index_uses_image_captioner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with contextlib.redirect_stdout(io.StringIO()):
                chunks = build_or_load_index(
                    "demo.pdf",
                    cache_path=str(temp_path / "demo.pdf.chunks.json"),
                    image_dir=str(temp_path / "images"),
                    caption_images=True,
                    image_captioner=lambda _path, _page: "Vue 首页标题图",
                )

            image_chunks = [
                chunk for chunk in chunks if chunk.kind == "image"
            ]
            self.assertEqual(len(image_chunks), 1)
            self.assertEqual(image_chunks[0].caption, "Vue 首页标题图")

    def test_build_chunks_falls_back_when_image_captioner_fails(
        self,
    ) -> None:
        def failing_captioner(_path: str, _page: int) -> str:
            raise ConnectionError("network unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            with contextlib.redirect_stdout(io.StringIO()):
                chunks = build_chunks(
                    "demo.pdf",
                    image_dir=temp_dir,
                    image_captioner=failing_captioner,
                )

            image_chunk = next(
                chunk for chunk in chunks if chunk.kind == "image"
            )
            self.assertEqual(image_chunk.caption, "第 1 页图片")

    def test_build_or_load_index_does_not_cache_failed_captions(
        self,
    ) -> None:
        def failing_captioner(_path: str, _page: int) -> str:
            raise ConnectionError("network unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "demo.pdf.chunks.json"

            with contextlib.redirect_stdout(io.StringIO()):
                chunks = build_or_load_index(
                    "demo.pdf",
                    cache_path=str(cache_path),
                    image_dir=str(Path(temp_dir) / "images"),
                    caption_images=True,
                    image_captioner=failing_captioner,
                )

            self.assertTrue(
                any(chunk.kind == "image" for chunk in chunks)
            )
            self.assertFalse(cache_path.exists())

    def test_describe_image_sends_image_to_vision_model(self) -> None:
        class FakeCompletions:
            def __init__(self) -> None:
                self.kwargs: dict | None = None

            def create(self, **kwargs: dict) -> object:
                self.kwargs = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="Vue 首页标题图"
                            )
                        )
                    ]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "demo.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nimage")
            completions = FakeCompletions()
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            )

            caption = describe_image(
                client,
                "qwen-vl-test",
                str(image_path),
                page=1,
            )

            self.assertEqual(caption, "Vue 首页标题图")
            self.assertIsNotNone(completions.kwargs)
            self.assertEqual(
                completions.kwargs["model"],
                "qwen-vl-test",
            )
            content = completions.kwargs["messages"][1]["content"]
            self.assertEqual(content[1]["type"], "image_url")
            self.assertTrue(
                content[1]["image_url"]["url"].startswith(
                    "data:image/png;base64,"
                )
            )

    def test_ask_qwen_uses_vision_model_for_image_chunk(self) -> None:
        class FakeCompletions:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs: dict) -> object:
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="图片标题是 The Progressive JavaScript Framework。[第 1 页]"
                            )
                        )
                    ]
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "demo.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nimage")
            completions = FakeCompletions()
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            )
            image_chunk = Chunk(
                chunk_id="page-1-image-0",
                source="demo.pdf",
                page=1,
                text="Vue 首页图片",
                terms=["vue", "图片"],
                kind="image",
                image_path=str(image_path),
                caption="Vue 首页图片",
            )

            with patch(
                "main.create_client",
                return_value=(
                    client,
                    "text-model",
                    "vision-model",
                ),
            ):
                answer = ask_qwen(
                    "图片中的标题是什么？",
                    [(9.0, image_chunk)],
                )

            self.assertIn(
                "The Progressive JavaScript Framework",
                answer,
            )
            self.assertEqual(
                completions.calls[0]["model"],
                "vision-model",
            )
            user_content = completions.calls[0]["messages"][1][
                "content"
            ]
            self.assertIsInstance(user_content, list)
            self.assertEqual(user_content[1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
