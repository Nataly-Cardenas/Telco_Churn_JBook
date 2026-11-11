"""Corrige un error de docutils 0.18 que rompe el diseño de la página con la bibliografía.

En docutils 0.18, Node.previous_sibling() calcula parent[index - 1]; para el primer hijo eso
es parent[-1], el último, en vez de None. El escritor HTML usa esa función para decidir si abre
el <div class="citation-list"> antes de la primera referencia. Como la primera referencia de
{bibliography} es el primer hijo de su contenedor, el <div> no se abre pero sí se cierra, y ese
</div> de más saca del artículo la navegación entre capítulos y el índice de la página.
docutils 0.19 corrigió la función; jupyter-book 0.15 exige docutils < 0.19, así que se corrige aquí.
"""
from docutils import nodes


def _previous_sibling(self):
    if self.parent is None:
        return None
    i = self.parent.index(self)
    return self.parent[i - 1] if i > 0 else None


def setup(app):
    nodes.Node.previous_sibling = _previous_sibling
    return {"parallel_read_safe": True, "parallel_write_safe": True}
