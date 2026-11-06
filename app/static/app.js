import { api, el, panelCard } from './common.js';

const $ = id => document.getElementById(id);
let queryUrl = null;
let browseOffset = 0;
let browseFilters = null;

function filters() {
  const params = new URLSearchParams();
  for (const [key, id] of [['dataset_id', 'dataset-filter'], ['participant_count', 'count-filter'], ['gender', 'gender-filter']]) {
    if ($(id).value !== '') params.set(key, $(id).value);
  }
  return params;
}

function busy(value) {
  for (const id of ['search-button', 'browse-button', 'more-button']) $(id).disabled = value;
}

function showError(error) {
  $('error').textContent = error.message;
  $('error').classList.remove('hidden');
}

async function browse(append = false) {
  if (!$('search-form').reportValidity()) return;
  busy(true); $('error').classList.add('hidden');
  $('search-status').textContent = 'Finding pose suggestions…';
  try {
    if (!append) { browseOffset = 0; browseFilters = filters(); }
    const params = new URLSearchParams(browseFilters);
    params.set('status', 'accepted'); params.set('limit', '24'); params.set('offset', browseOffset);
    const data = await api(`/api/panels?${params}`);
    if (!append) $('match-list').replaceChildren();
    data.panels.forEach(panel => $('match-list').append(panelCard(panel)));
    browseOffset += data.panels.length;
    $('results-heading').textContent = 'Pose suggestions';
    $('rank-note').classList.add('hidden');
    $('results-section').classList.remove('hidden');
    $('more-button').classList.toggle('hidden', browseOffset >= data.total);
    $('search-status').textContent = data.total ? `${data.total} accepted panel(s) fit your filters. Pick one for inspiration.` : 'No accepted panels fit these filters. Try fewer filters or accept panels in My library.';
  } catch (error) { showError(error); $('search-status').textContent = ''; }
  finally { busy(false); }
}

$('search-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!$('image-input').files.length) {
    $('search-status').textContent = 'Choose a photo before matching, or use Suggest poses to browse without one.';
    showError(new Error('No photo selected.'));
    $('image-input').focus();
    return;
  }
  busy(true); $('error').classList.add('hidden'); $('more-button').classList.add('hidden');
  $('results-section').classList.add('hidden');
  $('search-status').textContent = 'Detecting your pose and comparing figures…';
  try {
    const body = new FormData();
    body.append('image', $('image-input').files[0]);
    for (const [key, value] of filters()) body.append(key, value);
    const data = await api('/api/search', { method: 'POST', body });
    $('match-list').replaceChildren(...data.matches.map(match => panelCard(match)));
    $('results-heading').textContent = 'Closest panel matches';
    $('rank-note').classList.remove('hidden');
    $('results-section').classList.remove('hidden');
    $('search-status').textContent = data.matches.length
      ? `${data.matches.length} closest panel(s). Each compares an individual figure, including figures in group panels.`
      : 'No matches found. Check that your collection has accepted panels, loosen the filters, or choose a photo showing more of the body.';
  } catch (error) { showError(error); $('search-status').textContent = ''; }
  finally { busy(false); }
});

$('image-input').addEventListener('change', () => {
  $('error').classList.add('hidden');
  if (queryUrl) URL.revokeObjectURL(queryUrl);
  const file = $('image-input').files[0];
  queryUrl = file ? URL.createObjectURL(file) : null;
  $('query-preview').classList.toggle('hidden', !file);
  if (file) $('query-preview').src = queryUrl;
  else $('query-preview').removeAttribute('src');
  $('search-status').textContent = file ? 'Photo ready. Choose Match my pose to search accepted panels.' : '';
});
$('browse-button').addEventListener('click', () => browse());
$('more-button').addEventListener('click', () => browse(true));
window.addEventListener('beforeunload', () => { if (queryUrl) URL.revokeObjectURL(queryUrl); });

try {
  const data = await api('/api/datasets');
  data.datasets.forEach(dataset => {
    const option = el('option', dataset.name); option.value = dataset.id; $('dataset-filter').append(option);
  });
  $('empty-library').classList.toggle('hidden', data.datasets.some(dataset => dataset.accepted_count > 0));
} catch (error) { showError(error); }
