import { api, el, genderLabels, jsonOptions, panelCard } from './common.js';

const $ = id => document.getElementById(id);
const activeStatuses = new Set(['queued', 'running', 'cancelling']);
const supported = /\.(pdf|zip|cbz|jpg|jpeg|png|webp|bmp|gif|tif|tiff)$/i;
const jointThreshold = .35;
const bodyJoints = [11,12,13,14,15,16,23,24,25,26,27,28];
let datasets = [], selectedDataset = '', editingDataset = null, offset = 0, total = 0, pollTimer;
let datasetRequest = 0, panelRequest = 0;
let editor = null, pageImage = null, poseDirty = false, drawCrop = false, pointer = null;
const edges = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[27,31],[24,26],[26,28],[28,30],[30,32],[28,32]];
const jointNames = ['Nose','Left eye inner','Left eye','Left eye outer','Right eye inner','Right eye','Right eye outer','Left ear','Right ear','Left mouth','Right mouth','Left shoulder','Right shoulder','Left elbow','Right elbow','Left wrist','Right wrist','Left pinky','Right pinky','Left index','Right index','Left thumb','Right thumb','Left hip','Right hip','Left knee','Right knee','Left ankle','Right ankle','Left heel','Right heel','Left foot','Right foot'];
jointNames.forEach((name, index) => { const option = el('option', `${index}: ${name}`); option.value = index; $('joint-select').append(option); });

function error(message, review = false) {
  const target = $(review ? 'review-error' : 'error');
  target.textContent = message; target.classList.remove('hidden');
}
function clearError(review = false) { $(review ? 'review-error' : 'error').classList.add('hidden'); }
function run(action, review = false) { return Promise.resolve().then(action).catch(exc => error(exc.message, review)); }
function currentDataset() { return datasets.find(dataset => dataset.id === selectedDataset); }

function selectDataset(id) {
  if (id !== selectedDataset) {
    $('panel-list').replaceChildren();
    $('panel-summary').textContent = 'Loading panels…';
    panelRequest++;
  }
  selectedDataset = id;
}

async function loadDatasets(prefer = selectedDataset) {
  const sequence = ++datasetRequest;
  selectDataset(prefer);
  const response = await api('/api/datasets');
  if (sequence !== datasetRequest) return;
  datasets = response.datasets;
  selectDataset(datasets.some(dataset => dataset.id === selectedDataset) ? selectedDataset : datasets[0]?.id || '');
  $('dataset-select').replaceChildren();
  if (!datasets.length) { const option = el('option', 'Create a collection to begin'); option.value = ''; $('dataset-select').append(option); }
  datasets.forEach(dataset => { const option = el('option', dataset.name); option.value = dataset.id; $('dataset-select').append(option); });
  $('dataset-select').value = selectedDataset;
  const dataset = currentDataset();
  $('dataset-summary').textContent = dataset ? `${dataset.page_count} pages · ${dataset.accepted_count} accepted · ${dataset.pending_count} to review` : 'Your library starts empty. Add your own sources.';
  for (const id of ['import-section', 'dataset-actions', 'review-section']) $(id).classList.toggle('hidden', !dataset);
}

function datasetForm(dataset = null) {
  editingDataset = dataset?.id || null;
  $('dataset-form-title').textContent = dataset ? 'Collection details' : 'New collection';
  for (const name of ['name', 'source_url', 'license', 'attribution', 'notes']) $('dataset-form').elements[name].value = dataset?.[name] || '';
  $('dataset-form').classList.remove('hidden');
  $('dataset-form').elements.name.focus();
}
$('new-dataset').addEventListener('click', () => datasetForm());
$('edit-dataset').addEventListener('click', () => datasetForm(currentDataset()));
$('cancel-dataset').addEventListener('click', () => $('dataset-form').classList.add('hidden'));
$('dataset-form').addEventListener('submit', event => {
  event.preventDefault();
  run(async () => {
    clearError();
    const values = Object.fromEntries(new FormData(event.target));
    const dataset = await api(editingDataset ? `/api/datasets/${editingDataset}` : '/api/datasets', jsonOptions(editingDataset ? 'PATCH' : 'POST', values));
    $('dataset-form').classList.add('hidden');
    offset = 0; await loadDatasets(dataset.id); await loadPanels(); await loadJobs();
  });
});
$('dataset-select').addEventListener('change', () => run(async () => {
  selectDataset($('dataset-select').value); offset = 0;
  $('dataset-form').classList.add('hidden');
  await loadDatasets(); await loadPanels(); await loadJobs();
}));
$('delete-dataset').addEventListener('click', () => run(async () => {
  if (!confirm(`Delete “${currentDataset().name}” and its imported pages? Your original source files will be kept.`)) return;
  clearError(); await api(`/api/datasets/${selectedDataset}`, { method: 'DELETE' });
  offset = 0; await loadDatasets(); await loadPanels(); await loadJobs();
}));

function selectedFiles() {
  return [...$('files-input').files, ...$('folder-input').files].filter(file => supported.test(file.name));
}
for (const id of ['files-input', 'folder-input']) $(id).addEventListener('change', () => {
  // Each chooser represents one selection to avoid accidentally importing a previous batch twice.
  $(id === 'files-input' ? 'folder-input' : 'files-input').value = '';
  $('selected-files').textContent = `${selectedFiles().length} supported file(s) selected.`;
});
$('import-form').addEventListener('submit', event => {
  event.preventDefault();
  run(async () => {
    clearError(); const files = selectedFiles();
    if (!files.length) throw new Error('Choose PDFs, page images, or an archive first.');
    const body = new FormData();
    files.forEach(file => body.append('files', file, file.webkitRelativePath || file.name));
    $('import-button').disabled = true;
    try {
      await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `/api/datasets/${selectedDataset}/imports`);
        xhr.upload.onprogress = progress => { $('upload-status').textContent = progress.lengthComputable ? `Sending files to the local app: ${Math.round(progress.loaded / progress.total * 100)}%` : 'Sending files to the local app…'; };
        xhr.onerror = () => reject(new Error('Could not reach the local app.'));
        xhr.onload = () => {
          let data;
          try { data = JSON.parse(xhr.responseText); } catch { reject(new Error('Import request failed.')); return; }
          if (xhr.status >= 200 && xhr.status < 300) resolve(data);
          else reject(new Error(typeof data.detail === 'string' ? data.detail : 'Check the chosen files and try again.'));
        };
        xhr.send(body);
      });
      $('files-input').value = ''; $('folder-input').value = ''; $('selected-files').textContent = '';
      $('upload-status').textContent = 'Import queued. You can review completed pages while the rest are processed.';
      await loadJobs();
    } finally { $('import-button').disabled = false; }
  });
});

$('redetect-collection').addEventListener('click', () => run(async () => {
  if (!confirm('Replace automatic skeletons in unreviewed panels using the improved drawing detector? Accepted, skipped, tagged, and newly corrected panels are kept. Previously edited pending panels without tags or notes may also be replaced. A local catalog backup is saved first.')) return;
  clearError(); $('redetect-collection').disabled = true;
  $('upload-status').textContent = 'Preparing the drawing detector…';
  try {
    await api(`/api/datasets/${selectedDataset}/redetect`, { method: 'POST' });
    $('upload-status').textContent = 'Re-detection queued. Existing crops are reused; completed panels are saved as the job progresses.';
    await loadJobs();
  } finally { $('redetect-collection').disabled = false; }
}));

async function loadJobs() {
  clearTimeout(pollTimer);
  if (!selectedDataset) { $('jobs-section').classList.add('hidden'); return; }
  const forDataset = selectedDataset;
  const jobs = (await api(`/api/imports?dataset_id=${encodeURIComponent(forDataset)}`)).jobs;
  if (forDataset !== selectedDataset) return;
  $('jobs-section').classList.toggle('hidden', !jobs.length);
  $('jobs-list').replaceChildren();
  jobs.forEach(job => {
    const item = el('div', undefined, 'job');
    const redoing = job.kind === 'redetect';
    item.append(el('strong', `${redoing ? 'Re-detection · ' : ''}${job.status[0].toUpperCase() + job.status.slice(1)}`),
      el('p', redoing ? `${job.files_done}/${job.total_files} panels processed · ${job.panel_count} updated` : `${job.files_done}/${job.total_files} files · ${job.pages_done}/${job.total_pages} discovered pages · ${job.panel_count} panels`, 'small'));
    if (activeStatuses.has(job.status)) {
      const progress = el('progress'); progress.max = Math.max(1, job.total_files); progress.value = job.files_done;
      progress.setAttribute('aria-label', redoing ? 'Panels processed' : 'Import files completed'); item.append(progress);
      if (job.current) item.append(el('p', job.current, 'small source'));
      const cancel = el('button', job.status === 'cancelling' ? 'Stopping after current panel…' : 'Cancel import', 'secondary');
      cancel.disabled = job.status === 'cancelling';
      cancel.addEventListener('click', () => run(async () => { await api(`/api/imports/${job.id}/cancel`, { method: 'POST' }); await loadJobs(); }));
      item.append(cancel);
    }
    if (job.error) item.append(el('p', job.error, 'error'));
    if (job.status === 'cancelled') item.append(el('p', 'Completed pages are still available for review.', 'small muted'));
    if (job.warnings.length) {
      const details = el('details'); details.append(el('summary', `${job.warnings.length} recent warning(s)`));
      const list = el('div', undefined, 'warnings'); job.warnings.forEach(warning => list.append(el('p', warning))); details.append(list); item.append(details);
    }
    $('jobs-list').append(item);
  });
  if (jobs.some(job => activeStatuses.has(job.status))) {
    pollTimer = setTimeout(() => run(async () => { await loadDatasets(); await loadJobs(); if (!$('review-dialog').open) await loadPanels(); }), 1500);
  } else { await loadDatasets(); await loadPanels(); }
}

async function loadPanels() {
  if (!selectedDataset) { $('panel-list').replaceChildren(); return; }
  const forDataset = selectedDataset;
  const sequence = ++panelRequest;
  const params = new URLSearchParams({ dataset_id: forDataset, status: $('status-filter').value, limit: '24', offset: String(offset) });
  const data = await api(`/api/panels?${params}`);
  if (forDataset !== selectedDataset || sequence !== panelRequest) return;
  total = data.total;
  if (offset >= total && offset > 0) { offset = Math.max(0, Math.floor((total - 1) / 24) * 24); return loadPanels(); }
  $('panel-list').replaceChildren(...data.panels.map(panel => panelCard(panel, id => run(() => openPanel(id)))));
  $('panel-summary').textContent = total ? `${offset + 1}–${Math.min(offset + 24, total)} of ${total} panels` : 'No panels in this view. Import pages or choose a different status.';
  $('prev-page').disabled = offset === 0; $('next-page').disabled = offset + 24 >= total;
}
$('status-filter').addEventListener('change', () => run(async () => { offset = 0; $('panel-list').replaceChildren(); await loadPanels(); }));
$('prev-page').addEventListener('click', () => run(async () => { offset = Math.max(0, offset - 24); await loadPanels(); }));
$('next-page').addEventListener('click', () => run(async () => { offset += 24; await loadPanels(); }));

function cropBounds() { return ['left','top','right','bottom'].map(key => Number($(`crop-${key}`).value)); }
function setBounds(bounds) { ['left','top','right','bottom'].forEach((key, index) => { $(`crop-${key}`).value = bounds[index]; }); }
function selectedPose() { return editor?.poses[Number($('pose-select').value)]; }

async function openPanel(id) {
  clearError(true);
  const panel = await api(`/api/panels/${id}`);
  const image = new Image();
  await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error('Source page could not be loaded.')); image.src = panel.page_url; });
  editor = panel; pageImage = image; poseDirty = false; drawCrop = false;
  $('show-uncertain').checked = false;
  $('draw-crop').textContent = 'Draw crop';
  $('review-source').textContent = `${panel.dataset_name} · ${panel.source_name} · page ${panel.page_number} · ${panel.status}`;
  setBounds(panel.bounds); $('participant-count').value = panel.participant_count; $('panel-notes').value = panel.notes;
  renderGenders(panel.genders); renderPoses();
  $('review-canvas').width = panel.width; $('review-canvas').height = panel.height;
  draw();
  if (!$('review-dialog').open) $('review-dialog').showModal();
}

function renderGenders(values) {
  $('gender-tags').replaceChildren();
  const count = Math.min(64, Math.max(0, Number($('participant-count').value) || 0));
  for (let index = 0; index < count; index++) {
    const label = el('label', `Character ${index + 1}`); const select = el('select');
    for (const [value, title] of Object.entries(genderLabels)) { const option = el('option', title); option.value = value; select.append(option); }
    select.value = values[index] || 'unknown'; label.append(select); $('gender-tags').append(label);
  }
}
$('participant-count').addEventListener('input', () => renderGenders([...document.querySelectorAll('#gender-tags select')].map(select => select.value)));

function renderPoses(prefer = 0) {
  $('pose-select').replaceChildren();
  editor.poses.forEach((pose, index) => {
    const visible = bodyJoints.filter(joint => (pose.joint_scores?.[joint] ?? 1) >= jointThreshold).length;
    const option = el('option', `Figure ${index + 1}${pose.joints ? ` · ${visible}/12 body joints` : ' (re-detect to edit)'}`);
    option.value = index; $('pose-select').append(option);
  });
  if (!editor.poses.length) { const option = el('option', 'No skeletons detected'); option.value = ''; $('pose-select').append(option); }
  else $('pose-select').value = Math.min(prefer, editor.poses.length - 1);
  $('remove-pose').disabled = !editor.poses.length;
  $('editor-hint').textContent = editor.poses.some(pose => !pose.joints)
    ? 'Migrated vectors are preserved. Re-detect to get editable joints, or remove the old skeleton and add one manually.'
    : editor.poses.some(pose => pose.backend === 'legacy' || pose.backend === 'mediapipe-photo')
      ? 'These results used the older detector. Re-detect this panel to replace them, or correct the joints manually.'
      : 'Only confident joints are shown. Drag to correct a joint; Show uncertain joints reveals hidden estimates. Crop coordinates are editable.';
  syncJointFields();
}

function syncJointFields() {
  const joints = selectedPose()?.joints;
  const joint = joints?.[Number($('joint-select').value)];
  $('joint-x').value = joint?.[0] ?? ''; $('joint-y').value = joint?.[1] ?? '';
  const score = selectedPose()?.joint_scores?.[Number($('joint-select').value)];
  $('joint-confidence').textContent = score === undefined ? '' : `Joint score: ${score.toFixed(2)}. Setting coordinates marks this joint as trusted.`;
  $('set-joint').disabled = !joint;
}
$('pose-select').addEventListener('change', () => { syncJointFields(); draw(); });
$('joint-select').addEventListener('change', syncJointFields);
$('set-joint').addEventListener('click', () => {
  if (!selectedPose()?.joints) return;
  const x = Number($('joint-x').value), y = Number($('joint-y').value);
  if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x > editor.width || y > editor.height) { error('Joint coordinates must be inside the page.', true); return; }
  selectedPose().joints[Number($('joint-select').value)] = [x, y]; poseDirty = true;
  selectedPose().joint_scores ??= Array(33).fill(1);
  selectedPose().joint_scores[Number($('joint-select').value)] = 1;
  renderPoses(Number($('pose-select').value)); draw();
});
$('show-uncertain').addEventListener('change', draw);

function draw() {
  if (!pageImage || !editor) return;
  const canvas = $('review-canvas'), ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(pageImage, 0, 0, canvas.width, canvas.height);
  const [left, top, right, bottom] = cropBounds();
  ctx.fillStyle = '#18102355';
  ctx.fillRect(0,0,canvas.width,top); ctx.fillRect(0,bottom,canvas.width,canvas.height-bottom);
  ctx.fillRect(0,top,left,bottom-top); ctx.fillRect(right,top,canvas.width-right,bottom-top);
  ctx.strokeStyle = '#d32d66'; ctx.lineWidth = Math.max(3, canvas.width / 220);
  ctx.strokeRect(left, top, right - left, bottom - top);
  const radius = Math.max(4, canvas.width / 140);
  editor.poses.forEach((pose, index) => {
    if (!pose.joints) return;
    ctx.strokeStyle = index === Number($('pose-select').value) ? '#00b7a4' : '#7961d1';
    ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = Math.max(2, canvas.width / 300);
    const visible = joint => $('show-uncertain').checked || (pose.joint_scores?.[joint] ?? 1) >= jointThreshold;
    edges.forEach(([first, second]) => { if (!visible(first) || !visible(second)) return; ctx.beginPath(); ctx.moveTo(...pose.joints[first]); ctx.lineTo(...pose.joints[second]); ctx.stroke(); });
    pose.joints.forEach((joint, jointIndex) => { if (!visible(jointIndex)) return; ctx.beginPath(); ctx.arc(joint[0], joint[1], radius, 0, Math.PI * 2); ctx.fill(); });
  });
}
for (const key of ['left','top','right','bottom']) $(`crop-${key}`).addEventListener('input', draw);
$('draw-crop').addEventListener('click', () => {
  drawCrop = !drawCrop;
  $('draw-crop').textContent = drawCrop ? 'Cancel drawing' : 'Draw crop';
  $('editor-hint').textContent = drawCrop ? 'Drag from one corner of the panel to the opposite corner.' : 'Drag a joint to correct it, or edit the crop coordinates.';
});

function pagePoint(event) {
  const rect = $('review-canvas').getBoundingClientRect();
  return [Math.max(0, Math.min(editor.width, (event.clientX - rect.left) / rect.width * editor.width)),
          Math.max(0, Math.min(editor.height, (event.clientY - rect.top) / rect.height * editor.height))];
}
$('review-canvas').addEventListener('pointerdown', event => {
  if (!editor) return;
  const point = pagePoint(event);
  if (drawCrop) pointer = { crop: true, start: point };
  else {
    let nearest = null, distance = editor.width / $('review-canvas').getBoundingClientRect().width * 15;
    editor.poses.forEach((pose, poseIndex) => pose.joints?.forEach((joint, jointIndex) => {
      if (!$('show-uncertain').checked && (pose.joint_scores?.[jointIndex] ?? 1) < jointThreshold) return;
      const candidate = Math.hypot(joint[0] - point[0], joint[1] - point[1]);
      if (candidate < distance) { distance = candidate; nearest = { poseIndex, jointIndex }; }
    }));
    if (!nearest) return;
    pointer = nearest; $('pose-select').value = nearest.poseIndex; $('joint-select').value = nearest.jointIndex;
  }
  $('review-canvas').setPointerCapture(event.pointerId);
});
$('review-canvas').addEventListener('pointermove', event => {
  if (!pointer) return;
  const point = pagePoint(event);
  if (pointer.crop) setBounds([Math.floor(Math.min(pointer.start[0], point[0])), Math.floor(Math.min(pointer.start[1], point[1])), Math.ceil(Math.max(pointer.start[0], point[0])), Math.ceil(Math.max(pointer.start[1], point[1]))]);
  else {
    const pose = editor.poses[pointer.poseIndex];
    pose.joints[pointer.jointIndex] = point; pose.joint_scores ??= Array(33).fill(1);
    pose.joint_scores[pointer.jointIndex] = 1; poseDirty = true; syncJointFields();
  }
  draw();
});
function endPointer() {
  if (pointer?.crop) { drawCrop = false; $('draw-crop').textContent = 'Draw crop'; }
  pointer = null;
}
$('review-canvas').addEventListener('pointerup', endPointer);
$('review-canvas').addEventListener('pointercancel', endPointer);

$('add-pose').addEventListener('click', () => {
  if (editor.poses.some(pose => !pose.joints)) { error('Re-detect or remove migrated skeletons before adding a new one.', true); return; }
  const template = [[.5,.12],[.48,.10],[.46,.10],[.44,.11],[.52,.10],[.54,.10],[.56,.11],[.42,.13],[.58,.13],[.47,.17],[.53,.17],[.35,.28],[.65,.28],[.25,.43],[.75,.43],[.2,.58],[.8,.58],[.18,.6],[.82,.6],[.19,.62],[.81,.62],[.21,.61],[.79,.61],[.42,.55],[.58,.55],[.4,.73],[.6,.73],[.38,.91],[.62,.91],[.37,.94],[.63,.94],[.33,.95],[.67,.95]];
  const [left, top, right, bottom] = cropBounds();
  editor.poses.push({ joints: template.map(([x,y]) => [left + x * (right-left), top + y * (bottom-top)]), joint_scores: Array(33).fill(1), backend: 'manual' });
  poseDirty = true;
  if (Number($('participant-count').value) < editor.poses.length) {
    $('participant-count').value = editor.poses.length;
    renderGenders([...document.querySelectorAll('#gender-tags select')].map(select => select.value));
  }
  renderPoses(editor.poses.length - 1); draw();
});
$('remove-pose').addEventListener('click', () => {
  if (!editor.poses.length) return;
  editor.poses.splice(Number($('pose-select').value), 1); poseDirty = true; renderPoses(); draw();
});

async function detectOrAdd(add = false) {
  clearError(true);
  if (!confirm(add ? 'Save this crop as a separate panel for review?' : 'Replace this panel’s skeletons and gender tags with new detections?')) return;
  const buttons = ['redetect','add-panel']; buttons.forEach(id => $(id).disabled = true);
  try {
    const result = await api(add ? `/api/pages/${editor.page_id}/panels` : `/api/panels/${editor.id}/redetect`, jsonOptions('POST', { bounds: cropBounds() }));
    await openPanel(result.id); await loadPanels();
  } finally { buttons.forEach(id => $(id).disabled = false); }
}
$('redetect').addEventListener('click', () => run(() => detectOrAdd(), true));
$('add-panel').addEventListener('click', () => run(() => detectOrAdd(true), true));

async function savePanel(status) {
  clearError(true);
  if (!$('review-form').reportValidity()) return;
  const body = { status, bounds: cropBounds(), participant_count: Number($('participant-count').value),
    genders: [...document.querySelectorAll('#gender-tags select')].map(select => select.value), notes: $('panel-notes').value };
  if (poseDirty) {
    if (editor.poses.some(pose => !pose.joints)) throw new Error('Re-detect or remove the remaining migrated skeletons before saving edited poses.');
    body.poses = editor.poses.map(pose => pose.joints);
    body.pose_scores = editor.poses.map(pose => pose.joint_scores || Array(33).fill(1));
  }
  ['accept-panel','save-panel','skip-panel'].forEach(id => $(id).disabled = true);
  try {
    await api(`/api/panels/${editor.id}`, jsonOptions('PATCH', body));
    $('review-dialog').close(); editor = null;
    await loadDatasets(); await loadPanels();
  } finally { ['accept-panel','save-panel','skip-panel'].forEach(id => $(id).disabled = false); }
}
$('review-form').addEventListener('submit', event => { event.preventDefault(); run(() => savePanel('accepted'), true); });
$('save-panel').addEventListener('click', () => run(() => savePanel(editor.status), true));
$('skip-panel').addEventListener('click', () => run(async () => {
  await api(`/api/panels/${editor.id}`, jsonOptions('PATCH', { status: 'skipped' }));
  $('review-dialog').close(); editor = null; await loadDatasets(); await loadPanels();
}, true));
$('close-review').addEventListener('click', () => $('review-dialog').close());
$('review-dialog').addEventListener('close', () => { editor = null; pageImage = null; pointer = null; });
window.addEventListener('beforeunload', () => clearTimeout(pollTimer));

await run(async () => {
  const params = new URLSearchParams(location.search);
  await loadDatasets(params.get('dataset'));
  await loadPanels(); await loadJobs();
  if (!datasets.length) datasetForm();
  if (params.get('panel')) await openPanel(params.get('panel'));
});
