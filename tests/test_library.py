import io
import json
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.catalog import Catalog
from app.main import create_app
from app.pose_detector import normalize_pose
from scripts.batch_upload import multipart


def image_bytes():
    image = Image.new("RGB", (240, 320), "white")
    ImageDraw.Draw(image).rectangle((5, 5, 235, 315), fill="#bbbbbb")
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def joints():
    return [[40 + i * 3, 30 + i * 6] for i in range(33)]


def fake_details(image, manga=False):
    coords = np.asarray(joints(), np.float32) / [240, 320]
    return [{"vector": normalize_pose(coords.copy()), "bounds": [.1, .1, .8, .9], "joints": coords.tolist()}]


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = create_app(self.root, warm_up=False)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.catalog = self.app.state.catalog
        self.dataset = self.client.post("/api/datasets", json={"name": "References", "source_url": "https://example.org/book", "license": "Test fixture", "attribution": "Test author"}).json()["id"]

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def wait(self, job_id):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = self.catalog.job(job_id)
            if job["status"] not in {"queued", "running", "cancelling"}:
                return job
            time.sleep(.02)
        self.fail("Import did not finish.")

    def upload(self, files):
        response = self.client.post(f"/api/datasets/{self.dataset}/imports", files=files)
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["id"]

    @patch("app.importing.extract_pose_details", side_effect=fake_details)
    def test_pdf_folder_and_archive_imports_keep_provenance_and_review_gate(self, _):
        pdf = io.BytesIO()
        page = Image.new("RGB", (240, 320), "gray")
        page.save(pdf, "PDF", save_all=True, append_images=[page])
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("pages/10.png", image_bytes())
            target.writestr("pages/2.png", image_bytes())
            target.writestr("../../escape.png", image_bytes())
            target.writestr("notes.txt", "ignored")
        job_id = self.upload([("files", ("volume.pdf", pdf.getvalue(), "application/pdf")),
                              ("files", ("book.cbz", archive.getvalue(), "application/zip")),
                              ("files", ("folder/page2.png", image_bytes(), "image/png")),
                              ("files", ("folder/page10.png", image_bytes(), "image/png"))])
        job = self.wait(job_id)
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(job["pages_done"], 7)
        self.assertEqual(job["total_pages"], 7)
        self.assertEqual(job["files_done"], 4)
        self.assertEqual(job["warnings"], [])
        self.assertFalse((self.root / "escape.png").exists())
        panels = self.catalog.list_panels(self.dataset, "pending")["panels"]
        self.assertGreaterEqual(len(panels), 7)
        self.assertEqual(self.catalog.list_panels(self.dataset, "accepted")["total"], 0)
        self.assertTrue(any(panel["source_name"] == "folder/page10.png" for panel in panels))
        self.assertTrue(any(panel["source_name"] == "volume.pdf" and panel["page_number"] == 2 for panel in panels))
        panel = panels[0]
        response = self.client.get(panel["image_url"])
        self.assertEqual(response.status_code, 200)
        self.assertGreater(Image.open(io.BytesIO(response.content)).width, 0)
        self.assertEqual(panel["license"], "Test fixture")

    @patch("app.importing.extract_pose_details", return_value=[])
    def test_missed_pose_can_be_added_corrected_tagged_and_indexed(self, _):
        job = self.wait(self.upload([("files", ("page.png", image_bytes(), "image/png"))]))
        self.assertEqual(job["status"], "completed")
        panel = self.catalog.list_panels(self.dataset, "pending")["panels"][0]
        path = f"/api/panels/{panel['id']}"
        self.assertEqual(self.client.patch(path, json={"status": "accepted"}).status_code, 422)
        response = self.client.patch(path, json={"poses": [joints()], "participant_count": 2, "genders": ["female", "unknown"], "status": "accepted", "notes": "Pose corrected manually"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["poses"]), 1)
        self.assertEqual(self.client.get("/api/panels?gender=female&participant_count=2").json()["total"], 1)
        self.assertEqual(self.client.get("/api/panels?gender=male").json()["total"], 0)
        self.assertEqual(self.client.get("/api/panels?gender=unknown").json()["total"], 1)
        self.assertEqual(self.client.patch(path, json={"genders": ["female"]}).status_code, 422)
        self.assertEqual(self.client.patch(path, json={"bounds": [10,10,0,20]}).status_code, 422)
        self.assertEqual(self.client.patch(path, json={"poses": [[[1,2]]]}).status_code, 422)
        self.assertEqual(self.client.patch(path, json={"status": "skipped"}).status_code, 200)
        self.assertEqual(self.client.get("/api/panels").json()["total"], 0)

    def test_search_ranks_figures_deduplicates_panels_and_does_not_save_queries(self):
        vector = np.arange(66, dtype=np.float32) / 100
        other = vector + 1
        image = Image.new("RGB", (240,320))
        ids = self.catalog.add_page(self.dataset, "groups.png", 1, image, [
            {"bounds": [0,0,240,320], "poses": [{"vector": vector}, {"vector": other}]},
            {"bounds": [0,0,120,320], "poses": [{"vector": vector + .1}]},
        ], "accepted")
        self.catalog.update_panel(ids[0], {"genders": ["female", "unknown"]})
        before = set(self.root.rglob("*"))
        with patch("app.main.extract_pose_detections", return_value=[(vector, (0,0,1,1))]):
            response = self.client.post("/api/search", files={"image": ("personal-photo.png", image_bytes(), "image/png")})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual([match["id"] for match in result["matches"]], ids)
            self.assertAlmostEqual(result["matches"][0]["distance"], 0)
            self.assertNotIn("uploaded_image", result)
            filtered = self.client.post("/api/search", data={"participant_count": "2", "gender": "female"}, files={"image": ("query.png", image_bytes(), "image/png")}).json()
            self.assertEqual([match["id"] for match in filtered["matches"]], [ids[0]])
            empty = self.client.post("/api/search", data={"gender": "male"}, files={"image": ("query.png", image_bytes(), "image/png")}).json()
            self.assertEqual(empty["matches"], [])
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertEqual(self.client.get("/images/personal-photo.png").status_code, 404)

    def test_cancellation_retains_completed_pages_cleans_staging_and_blocks_delete(self):
        entered, release = threading.Event(), threading.Event()
        calls = 0

        def blocking_details(image, manga=False):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                self.assertTrue(release.wait(10))
            return fake_details(image, manga)

        with patch("app.importing.extract_pose_details", side_effect=blocking_details):
            job_id = self.upload([("files", (f"{i}.png", image_bytes(), "image/png")) for i in range(3)])
            try:
                self.assertTrue(entered.wait(10))
                self.assertEqual(self.client.delete(f"/api/datasets/{self.dataset}").status_code, 422)
                self.assertEqual(self.client.post(f"/api/imports/{job_id}/cancel").status_code, 200)
            finally:
                release.set()
            job = self.wait(job_id)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["pages_done"], 1)
        self.assertEqual(len(self.catalog.list_panels(self.dataset, "pending")["panels"]), 1)
        # close() waits for final staging cleanup after terminal status is written.
        self.app.state.imports.executor.shutdown(wait=True)
        self.assertEqual(list((self.root / "staging").iterdir()), [])
        self.client.post(f"/api/imports/{job_id}/cancel")
        self.assertEqual(self.catalog.job(job_id)["status"], "cancelled")
        self.assertEqual(self.client.delete(f"/api/datasets/{self.dataset}").status_code, 200)
        self.assertEqual(list(self.catalog.assets.iterdir()), [])

    @patch("app.importing.extract_pose_details", return_value=[])
    def test_bad_files_warn_without_discarding_good_pages_and_redetect_add_panel(self, _):
        job = self.wait(self.upload([("files", ("broken.pdf", b"bad pdf", "application/pdf")), ("files", ("good.png", image_bytes(), "image/png"))]))
        self.assertEqual(job["status"], "completed")
        self.assertTrue(job["warnings"])
        panel = self.catalog.list_panels(self.dataset, "pending")["panels"][0]
        response = self.client.post(f"/api/panels/{panel['id']}/redetect", json={"bounds": [0,0,120,200]})
        self.assertEqual(response.status_code, 200)
        response = self.client.post(f"/api/pages/{panel['page_id']}/panels", json={"bounds": [120,0,240,200]})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.catalog.list_panels(self.dataset, "pending")["total"], 2)

    @patch("app.importing.extract_pose_details", return_value=[])
    def test_streaming_cli_multipart_has_unambiguous_http_framing(self, _):
        source = self.root / "source.png"
        source.write_bytes(image_bytes())
        body, headers = multipart([source], self.root)
        prepared = requests.Request("POST", "http://127.0.0.1:8000", data=body, headers=headers).prepare()
        self.assertNotIn("Transfer-Encoding", prepared.headers)
        content = b"".join(body)
        self.assertEqual(int(prepared.headers["Content-Length"]), len(content))
        response = self.client.post(f"/api/datasets/{self.dataset}/imports", content=content, headers=headers)
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(self.wait(response.json()["id"])["status"], "completed")

    def test_cross_origin_writes_and_validation(self):
        self.assertEqual(self.client.post("/api/datasets", json={"name": "bad"}, headers={"origin": "https://unrelated.example"}).status_code, 403)
        self.assertEqual(self.client.post("/api/datasets", json={"name": " "}).status_code, 422)
        self.assertEqual(self.client.get("/api/panels?gender=invalid").status_code, 422)
        self.assertEqual(self.client.get("/api/panels/missing").status_code, 404)
        for path in ("/", "/library", "/full-page"):
            self.assertEqual(self.client.get(path).status_code, 200)
        schema = self.client.get("/openapi.json").json()
        self.assertIn("requestBody", schema["paths"]["/api/datasets/{dataset_id}/imports"]["post"])
        self.assertEqual(self.client.get("/api/datasets").headers["cache-control"], "no-store")


class MigrationTests(unittest.TestCase):
    def test_sqlite_catalog_and_interrupted_import_survive_application_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with TestClient(create_app(root, warm_up=False)) as client:
                dataset = client.post("/api/datasets", json={"name": "Persisted references"}).json()["id"]
                catalog = client.app.state.catalog
                panel_id = catalog.add_page(dataset, "source.png", 1, Image.new("RGB", (240,320)),
                    [{"bounds": [0,0,240,320], "poses": [{"vector": np.zeros(66, dtype=np.float32)}]}], "accepted")[0]
                job_id = catalog.create_job(dataset, 1)
                catalog.update_job(job_id, status="running")
                orphan = catalog.root / "staging" / ("a" * 32)
                orphan.mkdir(); (orphan / "staged-upload").write_bytes(b"temporary")
            with TestClient(create_app(root, warm_up=False)) as client:
                self.assertEqual(client.get(f"/api/imports/{job_id}").json()["status"], "interrupted")
                self.assertEqual(client.get(f"/api/panels/{panel_id}/image").status_code, 200)
                self.assertEqual(client.get("/api/datasets").json()["datasets"][0]["accepted_count"], 1)
                self.assertFalse(orphan.exists())

    def test_legacy_vectors_grouping_and_idempotence_survive_dataset_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images" / "page.png").write_bytes(image_bytes())
            vectors = np.arange(132, dtype=np.float32).reshape(2,66)
            np.savez(root / "pose_index.npz", vectors=vectors, schema_version=1)
            (root / "db_files.json").write_text(json.dumps(["/images/page.png"] * 2))
            (root / "pose_records.json").write_text(json.dumps([{"panel_bounds": [0,0,240,320]}] * 2))
            catalog = Catalog(root); catalog.initialize(); catalog.initialize()
            self.assertEqual(len(catalog.datasets()), 1)
            panels = catalog.list_panels(status="accepted")["panels"]
            self.assertEqual(len(panels), 1)
            self.assertEqual(panels[0]["participant_count"], 2)
            panel = catalog.panel(panels[0]["id"])
            np.testing.assert_array_equal(panel["poses"][0]["vector"], vectors[0])
            catalog.delete_dataset("legacy"); catalog.initialize()
            self.assertEqual(catalog.datasets(), [])
            self.assertTrue((root / "pose_index.npz").exists())

    def test_failed_migration_is_atomic_and_restart_marks_jobs_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "pose_index.npz", vectors=np.zeros((1,66)), schema_version=1)
            (root / "db_files.json").write_text(json.dumps(["/images/missing.png"]))
            catalog = Catalog(root)
            with self.assertRaises(FileNotFoundError): catalog.initialize()
            self.assertEqual(catalog.datasets(), [])
            (root / "images").mkdir(); (root / "images" / "missing.png").write_bytes(image_bytes())
            catalog.initialize()
            job_id = catalog.create_job("legacy", 1)
            catalog.update_job(job_id, status="running")
            catalog.initialize()
            self.assertEqual(catalog.job(job_id)["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
