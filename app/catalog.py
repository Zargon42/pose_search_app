"""SQLite is the source of truth; only rendered source pages live on disk."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

POSE_SCHEMA_VERSION = 1
GENDERS = {"unknown", "male", "female", "nonbinary", "other"}


def uid():
    return uuid.uuid4().hex


class Catalog:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.assets = self.root / "pages"
        self.path = self.root / "catalog.sqlite3"

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        self.assets.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, source_url TEXT NOT NULL DEFAULT '',
                    license TEXT NOT NULL DEFAULT '', attribution TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS pages (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                    source_name TEXT NOT NULL, page_number INTEGER NOT NULL, image_name TEXT NOT NULL,
                    width INTEGER NOT NULL, height INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS panels (
                    id TEXT PRIMARY KEY, page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                    bounds TEXT NOT NULL, participant_count INTEGER NOT NULL,
                    genders TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', notes TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS poses (
                    id TEXT PRIMARY KEY, panel_id TEXT NOT NULL REFERENCES panels(id) ON DELETE CASCADE,
                    vector BLOB NOT NULL, joints TEXT, bounds TEXT, schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'queued', total_files INTEGER NOT NULL,
                    files_done INTEGER NOT NULL DEFAULT 0, pages_done INTEGER NOT NULL DEFAULT 0,
                    total_pages INTEGER NOT NULL DEFAULT 0, panel_count INTEGER NOT NULL DEFAULT 0,
                    current TEXT NOT NULL DEFAULT '', warnings TEXT NOT NULL DEFAULT '[]', error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS pages_dataset ON pages(dataset_id);
                CREATE INDEX IF NOT EXISTS panels_page_status ON panels(page_id, status);
                CREATE INDEX IF NOT EXISTS poses_panel ON poses(panel_id);
            """)
            db.execute("UPDATE jobs SET status='interrupted', error='Import interrupted by app shutdown. Import the remaining files again.' WHERE status IN ('queued', 'running', 'cancelling')")
        self.migrate_legacy()

    def create_dataset(self, name, **metadata):
        dataset_id = uid()
        with self.connect() as db:
            db.execute("INSERT INTO datasets(id,name,source_url,license,attribution,notes) VALUES(?,?,?,?,?,?)",
                       (dataset_id, name, *(metadata.get(key, "") for key in ("source_url", "license", "attribution", "notes"))))
        return dataset_id

    def datasets(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("""
                SELECT d.*, COUNT(DISTINCT pg.id) AS page_count, COUNT(p.id) AS panel_count,
                COALESCE(SUM(p.status='pending'),0) AS pending_count,
                COALESCE(SUM(p.status='accepted'),0) AS accepted_count
                FROM datasets d LEFT JOIN pages pg ON pg.dataset_id=d.id
                LEFT JOIN panels p ON p.page_id=pg.id GROUP BY d.id ORDER BY d.created_at,d.id
            """)]

    def dataset(self, dataset_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
            if row is None:
                raise KeyError("Dataset not found.")
            return dict(row)

    def update_dataset(self, dataset_id, values):
        self.dataset(dataset_id)
        keys = [key for key in ("name", "source_url", "license", "attribution", "notes") if key in values]
        if keys:
            with self.connect() as db:
                db.execute(f"UPDATE datasets SET {','.join(key+'=?' for key in keys)} WHERE id=?",
                           (*[values[key] for key in keys], dataset_id))

    def delete_dataset(self, dataset_id):
        self.dataset(dataset_id)
        with self.connect() as db:
            if db.execute("SELECT 1 FROM jobs WHERE dataset_id=? AND status IN ('queued','running','cancelling')", (dataset_id,)).fetchone():
                raise ValueError("Cancel the active import and wait for it to stop before deleting this dataset.")
            names = [row[0] for row in db.execute("SELECT image_name FROM pages WHERE dataset_id=?", (dataset_id,))]
            db.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
        for name in names:
            (self.assets / name).unlink(missing_ok=True)

    def add_page(self, dataset_id, source_name, page_number, image, candidates, status="pending"):
        page_id = uid()
        filename = f"{page_id}.png"
        image.save(self.assets / filename, "PNG")
        panel_ids = []
        try:
            with self.connect() as db:
                db.execute("INSERT INTO pages VALUES(?,?,?,?,?,?,?)", (page_id, dataset_id, source_name, page_number, filename, image.width, image.height))
                for candidate in candidates:
                    panel_id = uid()
                    panel_ids.append(panel_id)
                    poses = candidate["poses"]
                    count = candidate.get("participant_count", len(poses))
                    db.execute("INSERT INTO panels(id,page_id,bounds,participant_count,genders,status) VALUES(?,?,?,?,?,?)",
                               (panel_id, page_id, json.dumps(candidate["bounds"]), count, json.dumps(["unknown"] * count), status))
                    self._insert_poses(db, panel_id, poses)
        except Exception:
            (self.assets / filename).unlink(missing_ok=True)
            raise
        return panel_ids

    @staticmethod
    def _insert_poses(db, panel_id, poses):
        for pose in poses:
            vector = np.asarray(pose["vector"], dtype=np.float32)
            if vector.shape != (66,) or not np.isfinite(vector).all():
                raise ValueError("Pose vectors must contain 66 finite coordinates.")
            db.execute("INSERT INTO poses VALUES(?,?,?,?,?,?)", (uid(), panel_id, vector.tobytes(),
                       json.dumps(pose.get("joints")), json.dumps(pose.get("bounds")), POSE_SCHEMA_VERSION))

    def add_panel(self, page_id, candidate):
        panel_id = uid()
        count = len(candidate["poses"])
        with self.connect() as db:
            db.execute("INSERT INTO panels(id,page_id,bounds,participant_count,genders) VALUES(?,?,?,?,?)",
                       (panel_id, page_id, json.dumps(candidate["bounds"]), count, json.dumps(["unknown"] * count)))
            self._insert_poses(db, panel_id, candidate["poses"])
        return panel_id

    @staticmethod
    def _panel(row):
        panel = dict(row)
        panel["bounds"] = json.loads(panel["bounds"])
        panel["genders"] = json.loads(panel["genders"])
        panel["image_url"] = f"/api/panels/{panel['id']}/image"
        panel["page_url"] = f"/api/pages/{panel['page_id']}/image"
        return panel

    PANEL_SELECT = """SELECT p.*, pg.dataset_id,pg.source_name,pg.page_number,pg.width,pg.height,
        d.name AS dataset_name,d.source_url,d.license,d.attribution
        FROM panels p JOIN pages pg ON pg.id=p.page_id JOIN datasets d ON d.id=pg.dataset_id"""

    def list_panels(self, dataset_id=None, status=None, participant_count=None, gender=None, limit=40, offset=0):
        clauses, args = [], []
        for column, value in (("pg.dataset_id", dataset_id), ("p.status", status), ("p.participant_count", participant_count)):
            if value is not None:
                clauses.append(column + "=?")
                args.append(value)
        if gender is not None:
            clauses.append("EXISTS(SELECT 1 FROM json_each(p.genders) WHERE value=?)")
            args.append(gender)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            total = db.execute("SELECT COUNT(*) FROM (" + self.PANEL_SELECT + where + ")", args).fetchone()[0]
            rows = db.execute(self.PANEL_SELECT + where + " ORDER BY pg.source_name,pg.page_number,pg.rowid,p.rowid LIMIT ? OFFSET ?", (*args, limit, offset))
            return {"total": total, "panels": [self._panel(row) for row in rows]}

    def panel(self, panel_id):
        with self.connect() as db:
            row = db.execute(self.PANEL_SELECT + " WHERE p.id=?", (panel_id,)).fetchone()
            if row is None:
                raise KeyError("Panel not found.")
            panel = self._panel(row)
            panel["poses"] = [{"id": pose["id"], "joints": json.loads(pose["joints"]), "bounds": json.loads(pose["bounds"]),
                               "vector": np.frombuffer(pose["vector"], dtype=np.float32).tolist()}
                              for pose in db.execute("SELECT * FROM poses WHERE panel_id=? ORDER BY rowid", (panel_id,))]
            return panel

    def page(self, page_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM pages WHERE id=?", (page_id,)).fetchone()
            if row is None:
                raise KeyError("Page not found.")
            return dict(row)

    def update_panel(self, panel_id, values, poses=None):
        with self.connect() as db:
            for key in ("bounds", "participant_count", "genders", "status", "notes"):
                if key in values:
                    value = json.dumps(values[key]) if key in ("bounds", "genders") else values[key]
                    db.execute(f"UPDATE panels SET {key}=? WHERE id=?", (value, panel_id))
            if poses is not None:
                db.execute("DELETE FROM poses WHERE panel_id=?", (panel_id,))
                self._insert_poses(db, panel_id, poses)

    def search(self, vectors, dataset_id=None, participant_count=None, gender=None, limit=12):
        # Rank figures, then keep only the closest figure from each panel.
        clauses = ["p.status='accepted'", "ps.schema_version=?"]
        args = [POSE_SCHEMA_VERSION]
        for column, value in (("pg.dataset_id", dataset_id), ("p.participant_count", participant_count)):
            if value is not None:
                clauses.append(column + "=?")
                args.append(value)
        if gender is not None:
            clauses.append("EXISTS(SELECT 1 FROM json_each(p.genders) WHERE value=?)")
            args.append(gender)
        best = {}
        with self.connect() as db:
            cursor = db.execute("SELECT ps.* FROM poses ps JOIN panels p ON p.id=ps.panel_id JOIN pages pg ON pg.id=p.page_id WHERE " + " AND ".join(clauses), args)
            while rows := cursor.fetchmany(4096):
                matrix = np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
                distances = np.stack([np.linalg.norm(matrix - vector, axis=1) for vector in vectors])
                query_indices = distances.argmin(axis=0)
                for index, row in enumerate(rows):
                    query_index = int(query_indices[index])
                    distance = float(distances[query_index, index])
                    panel_id = row["panel_id"]
                    if panel_id not in best or distance < best[panel_id]["distance"]:
                        best[panel_id] = {"id": panel_id, "distance": distance, "query_pose_index": query_index,
                                          "match_pose_bounds": json.loads(row["bounds"]), "pose_id": row["id"]}
        results = sorted(best.values(), key=lambda match: (match["distance"], match["id"]))[:limit]
        for match in results:
            panel = self.panel(match["id"])
            panel.pop("poses")
            match.update(panel)
        return results

    def create_job(self, dataset_id, count):
        job_id = uid()
        with self.connect() as db:
            db.execute("INSERT INTO jobs(id,dataset_id,total_files) VALUES(?,?,?)", (job_id, dataset_id, count))
        return job_id

    def jobs(self, dataset_id=None):
        with self.connect() as db:
            query = "SELECT * FROM jobs" + (" WHERE dataset_id=?" if dataset_id else "") + " ORDER BY rowid DESC LIMIT 100"
            return [self._job(row) for row in db.execute(query, (dataset_id,) if dataset_id else ())]

    @staticmethod
    def _job(row):
        result = dict(row)
        result["warnings"] = json.loads(result["warnings"])
        return result

    def job(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("Import not found.")
            return self._job(row)

    def update_job(self, job_id, **values):
        allowed = {"status", "files_done", "pages_done", "total_pages", "panel_count", "current", "warnings", "error"}
        if not values.keys() <= allowed:
            raise ValueError("Invalid job field.")
        with self.connect() as db:
            db.execute(f"UPDATE jobs SET {','.join(key+'=?' for key in values)} WHERE id=?",
                       (*[json.dumps(value) if key == "warnings" else value for key, value in values.items()], job_id))

    def migrate_legacy(self):
        with self.connect() as db:
            if db.execute("SELECT 1 FROM settings WHERE key='legacy_migrated'").fetchone():
                return
        index_path, files_path = self.root / "pose_index.npz", self.root / "db_files.json"
        if not index_path.exists() and not files_path.exists():
            return
        if not index_path.exists() or not files_path.exists():
            raise ValueError("Incomplete legacy index. Restore both pose_index.npz and db_files.json before migrating.")
        with np.load(index_path, allow_pickle=False) as data:
            vectors = data["vectors"].astype(np.float32)
            version = int(data["schema_version"]) if "schema_version" in data else 1
        paths = json.loads(files_path.read_text(encoding="utf-8"))
        records_path = self.root / "pose_records.json"
        records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else [{} for _ in paths]
        if vectors.shape != (len(paths), 66) or len(records) != len(paths) or version != POSE_SCHEMA_VERSION or not np.isfinite(vectors).all():
            raise ValueError("Legacy index has incompatible vectors or mismatched metadata; originals have been preserved.")
        grouped = {}
        for vector, path, record in zip(vectors, paths, records):
            if not isinstance(path, str) or not path.startswith("/images/") or Path(path).name != path[len("/images/"):]:
                raise ValueError("Legacy index contains an invalid image path.")
            grouped.setdefault(path, []).append((vector, record))
        # Everything, including the marker, commits together. Files have deterministic names for safe retries.
        with self.connect() as db:
            dataset_id = "legacy"
            if grouped:
                db.execute("INSERT INTO datasets(id,name,notes) VALUES(?,?,?)", (dataset_id, "Migrated collection", "Imported from the original local pose index; source rights unknown."))
            for path, entries in grouped.items():
                original = self.root / "images" / Path(path).name
                with Image.open(original) as image:
                    width, height = image.size
                    page_id = uuid.uuid5(uuid.NAMESPACE_URL, path).hex
                    filename = f"{page_id}.png"
                    image.convert("RGB").save(self.assets / filename, "PNG")
                db.execute("INSERT INTO pages VALUES(?,?,?,?,?,?,?)", (page_id, dataset_id, Path(path).name, 1, filename, width, height))
                panel_groups = {}
                for vector, record in entries:
                    bounds = record.get("panel_bounds") or [0, 0, width, height]
                    panel_groups.setdefault(tuple(bounds), []).append({"vector": vector, "bounds": record.get("pose_bounds"), "joints": None})
                for bounds, poses in panel_groups.items():
                    panel_id = uid()
                    db.execute("INSERT INTO panels(id,page_id,bounds,participant_count,genders,status) VALUES(?,?,?,?,?,'accepted')",
                               (panel_id, page_id, json.dumps(bounds), len(poses), json.dumps(["unknown"] * len(poses))))
                    self._insert_poses(db, panel_id, poses)
            db.execute("INSERT INTO settings VALUES('legacy_migrated','1')")
        # Preserve the old files as a backup. Only SQLite is used from now on.
