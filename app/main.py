from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

from app.src.shapes import Catenoid, Lattice

main_bp = Blueprint('main_bp', __name__,
                        template_folder='templates')

@main_bp.route('/', defaults={'page': 'index'})
@main_bp.route('/<page>')
def show(page):
    try:
        return render_template(f'{page}.html')
    except TemplateNotFound:
        abort(404)

shape_bp = Blueprint('shape_bp', __name__)

@shape_bp.route('/create-shape', methods=['POST'])
def create_shape(type, r, thickness, density=None):
    if type == 'catenoid':
        shape = Catenoid(r, thickness)
        return f'Created a catenoid with radius {r} and thickness {thickness}'
    elif type == 'lattice': 
        shape = Lattice(r, thickness, density)
        return f'Created a lattice with radius {r} and thickness {thickness}'
    else:
        return 'Shape type not supported', 400