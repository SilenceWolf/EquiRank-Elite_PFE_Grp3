/* ============================================================
   © 2026 PFE EquiRank Elite — Tous droits réservés.
   Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
           · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana.
   Licence : PROPRIETARY — voir fichier LICENSE.
   ============================================================ */
window.EquirankState = (function() {
  const PREFIX = 'equirank.';

  function get(key, fallback) {
    try {
      const raw = sessionStorage.getItem(PREFIX + key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch(e) { return fallback; }
  }

  function set(key, value) {
    try {
      sessionStorage.setItem(PREFIX + key, JSON.stringify(value));
    } catch(e) { /* quota / private mode → silencieux */ }
  }

  function clear(key) {
    try { sessionStorage.removeItem(PREFIX + key); } catch(e) {}
  }

  // Branche un input/select/checkbox sur sessionStorage : sa valeur est
  // restaurée au load et persistée à chaque change.
  function bindField(elId, key) {
    const el = document.getElementById(elId);
    if (!el) return;
    const saved = get(key, null);
    if (saved !== null && saved !== '') {
      // Pour les <select>, ne restaure que si l'option existe encore.
      if (el.tagName === 'SELECT') {
        const opt = Array.from(el.options).find(o => o.value === saved);
        if (opt) el.value = saved;
      } else {
        el.value = saved;
      }
    }
    el.addEventListener('change', () => set(key, el.value));
    if (el.tagName === 'INPUT') {
      el.addEventListener('input', () => set(key, el.value));
    }
  }

  return { get, set, clear, bindField };
})();
