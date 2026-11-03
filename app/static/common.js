export async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map(item => `${item.loc?.slice(-1)[0] || 'Input'}: ${item.msg}`).join('; ')
      : data.detail;
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return data;
}

export function jsonOptions(method, body) {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
}

export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

export const genderLabels = { unknown: 'Unknown', male: 'Male', female: 'Female', nonbinary: 'Nonbinary', other: 'Other' };

export function panelCard(panel, onOpen) {
  const card = el('article', undefined, 'panel-card');
  const figure = el('div', undefined, 'panel-figure');
  const image = el('img');
  image.src = panel.image_url;
  image.alt = `Panel from ${panel.source_name}, page ${panel.page_number}`;
  image.loading = 'lazy';
  figure.append(image);
  if (panel.match_pose_bounds) {
    const canvas = el('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    figure.append(canvas);
    image.addEventListener('load', () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d');
      const [x, y, right, bottom] = panel.match_pose_bounds;
      context.strokeStyle = '#7c3aed';
      context.lineWidth = Math.max(2, canvas.width / 150);
      context.strokeRect(x - panel.bounds[0], y - panel.bounds[1], right - x, bottom - y);
    });
  }
  const content = el('div', undefined, 'card-content');
  const line = el('div', undefined, 'card-line');
  line.append(el('span', `${panel.participant_count} ${panel.participant_count === 1 ? 'person' : 'people'}`, 'badge'));
  if (panel.distance !== undefined) line.append(el('span', `Distance ${panel.distance.toFixed(3)}`, 'muted small'));
  else line.append(el('span', panel.status, 'muted small'));
  content.append(line, el('h3', panel.dataset_name), el('p', `${panel.source_name} · page ${panel.page_number}`, 'source'));
  const known = panel.genders.filter(gender => gender !== 'unknown');
  if (known.length) content.append(el('p', known.map(gender => genderLabels[gender]).join(', '), 'small muted'));
  if (panel.attribution) content.append(el('p', panel.attribution, 'small muted'));
  if (panel.license) content.append(el('p', `License: ${panel.license}`, 'small muted'));
  if (panel.source_url) {
    try {
      const url = new URL(panel.source_url);
      if (['http:', 'https:'].includes(url.protocol)) {
        const source = el('a', 'Source website');
        source.href = url.href; source.target = '_blank'; source.rel = 'noreferrer';
        content.append(source);
      }
    } catch { /* Provenance can also be a local reference. */ }
  }
  if (onOpen) {
    const button = el('button', 'Review panel', 'secondary');
    button.type = 'button';
    button.addEventListener('click', () => onOpen(panel.id));
    content.append(button);
  } else {
    const link = el('a', 'Edit tags & skeleton', 'text-link');
    link.href = `/library?dataset=${encodeURIComponent(panel.dataset_id)}&panel=${encodeURIComponent(panel.id)}`;
    content.append(link);
  }
  card.append(figure, content);
  return card;
}
