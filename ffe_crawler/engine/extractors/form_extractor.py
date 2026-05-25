# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

import re
from bs4 import Tag
from .base_extractor import BaseCrawlerExtractor


class FormExtractor(BaseCrawlerExtractor):
    """
    Extrait la structure des formulaires : champs, labels, valeurs par défaut,
    options de select, et boutons.
    Utile pour mapper les formulaires d'une page sans les soumettre.
    """

    def can_handle(self, element: Tag) -> bool:
        return bool(
            element.find(['form', 'input', 'select', 'textarea'])
            or element.name in ('form', 'input', 'select', 'textarea')
        )

    def extract(self, element: Tag, url: str) -> list[dict]:
        rows: list[dict] = []

        forms = element.find_all('form') or ([element] if element.name == 'form' else [])
        # Si pas de <form> explicite, on cherche quand même les champs libres
        if not forms:
            forms = [element]

        for form_idx, form in enumerate(forms):
            form_action  = form.get('action', '') if form.name == 'form' else ''
            form_method  = form.get('method', 'GET').upper() if form.name == 'form' else ''
            form_context = f'form={form_idx} action={form_action} method={form_method}'

            # Champs <input>
            for inp in form.find_all('input'):
                rows.append({
                    'source_url': url,
                    'extractor':  self.name,
                    'type':       'form_field',
                    'tag':        'input',
                    'content':    inp.get('value', ''),
                    'extra': (
                        f'name={inp.get("name", "")} | '
                        f'type={inp.get("type", "text")} | '
                        f'placeholder={inp.get("placeholder", "")} | '
                        f'required={inp.has_attr("required")} | '
                        f'{form_context}'
                    ),
                })

            # Listes déroulantes <select>
            for sel in form.find_all('select'):
                options = [
                    f'{opt.get("value", opt.get_text(strip=True))}:{opt.get_text(strip=True)}'
                    for opt in sel.find_all('option')
                ]
                rows.append({
                    'source_url': url,
                    'extractor':  self.name,
                    'type':       'form_select',
                    'tag':        'select',
                    'content':    ' | '.join(options),
                    'extra': (
                        f'name={sel.get("name", "")} | '
                        f'multiple={sel.has_attr("multiple")} | '
                        f'{form_context}'
                    ),
                })

            # Zones de texte <textarea>
            for ta in form.find_all('textarea'):
                rows.append({
                    'source_url': url,
                    'extractor':  self.name,
                    'type':       'form_textarea',
                    'tag':        'textarea',
                    'content':    ta.get_text(strip=True),
                    'extra': (
                        f'name={ta.get("name", "")} | '
                        f'placeholder={ta.get("placeholder", "")} | '
                        f'{form_context}'
                    ),
                })

            # Boutons de soumission
            for btn in form.find_all(['button', 'input'], type=re.compile(r'submit|button|reset', re.I) if hasattr(form, 'find_all') else True):
                btn_type = btn.get('type', 'submit')
                if btn_type not in ('submit', 'button', 'reset'):
                    continue
                rows.append({
                    'source_url': url,
                    'extractor':  self.name,
                    'type':       'form_button',
                    'tag':        btn.name,
                    'content':    btn.get_text(strip=True) or btn.get('value', ''),
                    'extra':      f'type={btn_type} | {form_context}',
                })

        return rows
