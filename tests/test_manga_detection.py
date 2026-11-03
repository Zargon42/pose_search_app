import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.catalog import Catalog
from app.main import create_app
from app.pose_detector import MangaPoseDetector, BODY_JOINTS, halpe26_to_landmarks, usable_body
from app.pose_matching import body_distances


def body_coordinates():
    coords = np.full((33,2), .5, dtype=np.float32)
    coords[BODY_JOINTS] = [[.35,.25],[.65,.25],[.25,.4],[.75,.4],[.2,.55],[.8,.55],
                          [.4,.55],[.6,.55],[.4,.75],[.6,.75],[.4,.95],[.6,.95]]
    return coords


class DrawingDetectorTests(unittest.TestCase):
    def test_named_mapping_and_unsupported_joints_have_no_confidence(self):
        points = np.array([[index*3+1,index*2+2] for index in range(26)],np.float32)
        coords, scores = halpe26_to_landmarks(points, np.ones(26), 100, 100)
        np.testing.assert_allclose(coords[11], points[5]/100)
        np.testing.assert_allclose(coords[24], points[12]/100)
        np.testing.assert_allclose(coords[29], points[24]/100)
        self.assertEqual(scores[19], 0)  # RTMPose body model does not predict fingers.
        self.assertEqual(scores[1], 0)   # Nor MediaPipe's extra eye landmarks.
        self.assertEqual(scores[15], 1)
        self.assertEqual(scores[28], 1)

    def test_off_image_and_nonfinite_estimates_are_not_trusted(self):
        points = np.zeros((26,2),np.float32)
        points[5] = [-10, 50]
        points[6] = [float('nan'), 50]
        confidence = np.ones(26); confidence[7] = float('inf')
        coords, scores = halpe26_to_landmarks(points, confidence,100,100)
        self.assertEqual(scores[11],0); self.assertEqual(scores[12],0); self.assertEqual(scores[13],0)
        self.assertTrue(np.isfinite(coords).all())

    def test_no_detected_person_never_becomes_a_full_image_skeleton(self):
        detector = MangaPoseDetector()
        detector._detector = Mock(return_value=np.empty((0,4)))
        detector._pose_model = Mock()
        self.assertEqual(detector.detect_details(Image.new('RGB',(100,100))), [])
        detector._pose_model.assert_not_called()

    def test_face_only_or_sparse_body_is_rejected(self):
        coords = body_coordinates()
        scores = np.zeros(33); scores[:11]=1
        self.assertFalse(usable_body(coords,scores))
        scores[BODY_JOINTS[:3]]=1
        self.assertFalse(usable_body(coords,scores))
        scores[BODY_JOINTS]=1
        self.assertTrue(usable_body(coords,scores))

    def test_missing_models_fail_with_setup_instruction(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.pose_detector.MANGA_MODEL_DIR',Path(directory)):
            with self.assertRaisesRegex(RuntimeError, 'setup_pose_models'):
                MangaPoseDetector().detect_details(Image.new('RGB',(100,100)))


class ConfidenceMatchingTests(unittest.TestCase):
    def test_fingers_and_face_do_not_change_body_pose_matching(self):
        query = body_coordinates().flatten()
        changed = body_coordinates()
        changed[:11]=1000; changed[17:23]=-1000
        distance = body_distances(changed.reshape(1,66),query,np.ones((1,33)),np.ones(33))[0]
        self.assertAlmostEqual(distance,0,places=6)

    def test_hidden_legs_are_excluded_and_partial_matches_have_coverage_penalty(self):
        query = body_coordinates().flatten()
        candidate = body_coordinates()
        hidden = [25,26,27,28]
        candidate[hidden] = 1000
        scores = np.ones((1,33)); scores[:,hidden]=0
        distance = body_distances(candidate.reshape(1,66),query,scores,np.ones(33))[0]
        self.assertAlmostEqual(distance,.2*4/12,places=6)
        scores[:]=0
        self.assertTrue(np.isinf(body_distances(candidate.reshape(1,66),query,scores,np.ones(33))[0]))

    def test_pixel_aspect_ratio_scale_and_translation_are_handled(self):
        pixels = body_coordinates()*[200,200]
        candidate = (pixels/[400,200]).flatten()
        query = ((pixels*2+[40,50])/[800,600]).flatten()
        distance = body_distances(candidate.reshape(1,66),query,np.ones((1,33)),np.ones(33),
                                  dimensions=[[400,200]],query_dimensions=[800,600])[0]
        self.assertAlmostEqual(distance,0,places=6)


class RedetectionTests(unittest.TestCase):
    def test_inference_failure_is_reported_and_preserves_old_pose(self):
        import time
        with tempfile.TemporaryDirectory() as directory, TestClient(create_app(directory,warm_up=False)) as client:
            catalog=client.app.state.catalog; dataset=catalog.create_dataset('Test')
            panel_id=catalog.add_page(dataset,'page.png',1,Image.new('RGB',(200,200)),
                [{'bounds':[0,0,200,200],'poses':[{'vector':body_coordinates().flatten(),'backend':'legacy'}]}])[0]
            before=catalog.panel(panel_id)
            with patch('app.main._manga_pose_detector._load'),patch('app.importing.extract_pose_details',side_effect=RuntimeError('Model inference failed')):
                job_id=client.post(f'/api/datasets/{dataset}/redetect').json()['id']
                deadline=time.monotonic()+5
                while catalog.job(job_id)['status'] in {'queued','running','cancelling'} and time.monotonic()<deadline: time.sleep(.01)
            self.assertEqual(catalog.job(job_id)['status'],'failed')
            self.assertIn('inference failed',catalog.job(job_id)['warnings'][0])
            self.assertEqual(catalog.panel(panel_id),before)

    def test_reprocess_preserves_reviewed_tagged_manual_and_crop_records(self):
        import time
        with tempfile.TemporaryDirectory() as directory, TestClient(create_app(directory,warm_up=False)) as client:
            catalog = client.app.state.catalog
            dataset = catalog.create_dataset('Drawing collection')
            automatic = {'vector':body_coordinates().flatten(),'joints':body_coordinates().tolist(),'bounds':(.1,.1,.9,.9),
                         'joint_scores':[1.]*33,'backend':'legacy'}
            ids=catalog.add_page(dataset,'page.png',1,Image.new('RGB',(200,200)),
                                 [{'bounds':[0,0,200,200],'poses':[automatic]} for _ in range(7)])
            catalog.update_panel(ids[1],{'status':'accepted'})
            catalog.update_panel(ids[2],{'status':'skipped'})
            catalog.update_panel(ids[3],{'genders':['female']})
            catalog.update_panel(ids[4],{'notes':'review in progress'})
            with catalog.connect() as db: db.execute("UPDATE poses SET backend='manual' WHERE panel_id=?",(ids[5],))
            catalog.update_panel(ids[6],{'participant_count':2,'genders':['unknown','unknown']})
            before={panel_id:catalog.panel(panel_id) for panel_id in ids}
            def details(image,manga=False):
                return [{**automatic,'backend':'rtmpose-humanart'}]
            with patch('app.main._manga_pose_detector._load'),patch('app.importing.extract_pose_details',side_effect=details):
                response=client.post(f'/api/datasets/{dataset}/redetect')
                self.assertEqual(response.status_code,202,response.text)
                job_id=response.json()['id']
                deadline=time.monotonic()+5
                while catalog.job(job_id)['status'] in {'queued','running','cancelling'} and time.monotonic()<deadline: time.sleep(.01)
            job=catalog.job(job_id)
            self.assertEqual(job['status'],'completed',job)
            self.assertEqual(job['total_files'],1)
            self.assertEqual(job['panel_count'],1)
            self.assertEqual(job['kind'],'redetect')
            self.assertEqual(catalog.panel(ids[0])['poses'][0]['backend'],'rtmpose-humanart')
            self.assertEqual(catalog.panel(ids[0])['bounds'],before[ids[0]]['bounds'])
            for panel_id in ids[1:]: self.assertEqual(catalog.panel(panel_id),before[panel_id])
            backup=next((Path(directory)/'backups').glob('*.sqlite3'))
            self.assertGreater(backup.stat().st_size,0)

    def test_review_and_cancellation_during_inference_preserve_results(self):
        import time
        entered, release=threading.Event(),threading.Event()
        with tempfile.TemporaryDirectory() as directory, TestClient(create_app(directory,warm_up=False)) as client:
            catalog=client.app.state.catalog; dataset=catalog.create_dataset('Test')
            pose={'vector':body_coordinates().flatten(),'joints':body_coordinates().tolist(),'bounds':(.1,.1,.9,.9),'backend':'legacy'}
            ids=catalog.add_page(dataset,'page.png',1,Image.new('RGB',(200,200)),
                                 [{'bounds':[0,0,200,200],'poses':[pose]} for _ in range(3)])
            calls=0
            def details(image,manga=False):
                nonlocal calls
                calls+=1
                if calls==2:
                    entered.set(); release.wait(5)
                return [{**pose,'backend':'rtmpose-humanart'}]
            with patch('app.main._manga_pose_detector._load'),patch('app.importing.extract_pose_details',side_effect=details):
                job_id=client.post(f'/api/datasets/{dataset}/redetect').json()['id']
                try:
                    self.assertTrue(entered.wait(5))
                    before=catalog.panel(ids[1])['poses']
                    self.assertEqual(client.patch(f'/api/panels/{ids[1]}',json={'status':'accepted'}).status_code,200)
                    client.post(f'/api/imports/{job_id}/cancel')
                finally: release.set()
                deadline=time.monotonic()+5
                while catalog.job(job_id)['status'] in {'queued','running','cancelling'} and time.monotonic()<deadline: time.sleep(.01)
            self.assertEqual(catalog.job(job_id)['status'],'cancelled')
            self.assertEqual(catalog.job(job_id)['panel_count'],1)
            self.assertEqual(catalog.panel(ids[1])['status'],'accepted')
            self.assertEqual(catalog.panel(ids[1])['poses'],before)
            self.assertEqual(catalog.panel(ids[2])['poses'][0]['backend'],'legacy')

    def test_old_sqlite_catalog_gains_confidence_columns_without_vector_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog=Catalog(Path(directory)); catalog.initialize()
            dataset=catalog.create_dataset('Old collection')
            panel_id=catalog.add_page(dataset,'old.png',1,Image.new('RGB',(200,200)),
                [{'bounds':[0,0,200,200],'poses':[{'vector':body_coordinates().flatten()}]}])[0]
            expected=catalog.panel(panel_id)['poses'][0]['vector']
            with catalog.connect() as db:
                db.execute('ALTER TABLE poses DROP COLUMN joint_scores')
                db.execute('ALTER TABLE poses DROP COLUMN backend')
            catalog.initialize()
            pose=catalog.panel(panel_id)['poses'][0]
            self.assertEqual(pose['vector'],expected)
            self.assertEqual(pose['backend'],'legacy')
            self.assertEqual(pose['joint_scores'],[1.]*33)


if __name__=='__main__': unittest.main()
