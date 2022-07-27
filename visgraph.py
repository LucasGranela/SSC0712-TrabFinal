import pyvisgraph as vg
from pyvisgraph import Point

# polys = [[vg.Point(0.0,1.0), vg.Point(3.0,1.0), vg.Point(1.5,4.0)],
#           [vg.Point(4.0,4.0), vg.Point(7.0,4.0), vg.Point(5.5,8.0)]]

polys = [[vg.Point(0.0,1.0), vg.Point(3.0,1.0), vg.Point(1.5,4.0)],
         [vg.Point(1.5,2.0), vg.Point(7.0,2.0), vg.Point(6.0,4.0)]]

polys = [[Point(666.00, 794.00), Point(656.00, 794.00), Point(666.00, 487.00), Point(656.00, 487.00)], [Point(55.00, 40.00), Point(55.00, 30.00), Point(811.00, 40.00), Point(811.00, 30.00)]]

g = vg.VisGraph()
g.build(polys)
g.save('./graph.pk1')

shortest = g.shortest_path(vg.Point(1.5,0.0), vg.Point(4.0, 6.0))
print( shortest )

# g2 = vg.VisGraph()
# g2.load('./graph.pk1')