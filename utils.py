import lxml.html
import numpy as np
import requests
import tensorflow as tf

from IPython.display import clear_output, Image, display, HTML


def inspect(inspectable, init_global=False, init_table=False):
    with tf.Session() as sess:
        if init_global:
            sess.run(tf.global_variables_initializer())
        if init_table:
            sess.run(tf.tables_initializer())
        print(sess.run(inspectable))


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def
  
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def


def show_graph(graph_def=None, max_const_size=32, make_huge=False):
    """Visualize TensorFlow graph.
    stolen from: https://blog.jakuba.net/2017/05/30/tensorflow-visualization.html
    """
    graph_def = graph_def or tf.get_default_graph().as_graph_def()
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:{div_height_px}px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(
        data=repr(str(strip_def)), 
        id='graph' + str(np.random.rand()),
        div_height_px=1000 if make_huge else 600,
    )
  
    iframe = """
        <iframe seamless style="width:100%;height:{}px;border:0" srcdoc="{}"></iframe>
    """.format(
        1020 if make_huge else 620,
        code.replace('"', '&quot;')
    )
    display(HTML(iframe))


def _extremely_lazy_documentation_headings(url):
    resp = requests.get(url)
    root = lxml.html.fromstring(resp.text)
    headings = [
        '{} {}'.format(
            '##' if elem.tag == 'h2' else '###',
            elem.text_content().lower()
        )
        for elem in root.xpath('//h2|//h3')
    ]
    
    for h in headings:
        if h not in ['### stay connected', '### support']:
            print(h)
    print('# summary')