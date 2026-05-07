from src.explorer import DomainExplorer
ex = DomainExplorer()
text = 'LLM and GPT models are used in NLP; ASR uses end-to-end approaches.'
terms = ex._extract_terms_from_text(text)
found = {t.upper() for t in terms}
print('Terms:', terms)
print('Found:', found)
print('Check LLM/GPT/ASR/NLP:', any(x in found for x in ['LLM', 'GPT', 'ASR', 'NLP']))
