from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

main_bp = Blueprint('main_bp', __name__,
                        template_folder='templates')

@main_bp.route('/', defaults={'page': 'index'})
@main_bp.route('/<page>')
def show(page):
    try:
        return render_template(f'pages/{page}.html')
    except TemplateNotFound:
        abort(404)