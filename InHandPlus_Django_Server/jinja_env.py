from django.templatetags.static import static
from django.urls import reverse_lazy

from jinja2 import Environment


def environment(**options):
    env = Environment(**options)
    env.globals.update({
        'static': static,
        'url': reverse_lazy,
    })
    return env