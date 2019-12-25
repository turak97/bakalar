from bokeh.plotting import figure
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

def modify_doc(doc):
    p = figure()
    p.line([1,2,3,4,5], [3,4,2,7,5], line_width=2)
    doc.add_root(p)

if __name__ == '__main__':
    server = Server({'/bkapp': modify_doc}, io_loop=IOLoop())
    server.start()
    server.io_loop.start()