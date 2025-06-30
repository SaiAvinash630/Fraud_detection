# context_processors.py
from datetime import datetime

def inject_current_year():
    return {'current_year': datetime.now().year}
