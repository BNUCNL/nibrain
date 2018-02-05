# attributes class
class GeometryAttribute(object):
    def __init__(self, name, vertex_coords, vertex_faces, vertex_id):
        """
        Init GeometryAttribute.

        Parameters
        ----------
            name: the name of where geometry indicated, like 'inflated', 'sphere' etc.
            vertex_coords: coords of vertexes, should be N*3 array.
            vertex_faces: faces of vertexes, should be M*3 array.
            vertex_id: vertexes id in this geometry, should be K*1 array.
        """
        self.name = name
        self.vertex_coords = vertex_coords
        self.vertex_faces = vertex_faces
        self.vertex_id = vertex_id

    def get_name(self):
        return self.name

    def set_name(self, name):
        if not isinstance(name, str):
            raise ValueError("Input 'name' should be string.")
        self.name = name

    def get_vertex_coords(self):
        return self.vertex_coords

    def set_vertex_coords(self, vertex_coords):
        if vertex_coords.shape[1] != 3:
            raise ValueError("The shape of input should be [N, 3].")
        self.vertex_coords = vertex_coords

    def get_vertex_faces(self):
        return self.vertex_faces

    def set_vertex_faces(self, vertex_faces):
        if vertex_faces.shape[1] != 3:
            raise ValueError("The shape of input should be [N, 3].")
        self.vertex_faces = vertex_faces

    def get_vertex_id(self):
        return self.vertex_id

    def set_vertex_id(self, vertex_id):
        self.vertex_id = vertex_id


class ConnectionAttribute(object):
    def __init__(self):
        self.region = []
        self.tract = []

    def get_region(self):
        return self.region

    def set_region(self, region):
        self.region = region

    def get_tract(self):
        return self.tract

    def set_tract(self, tract):
        self.tract = tract

    def merge(self, ca):
        """
        Merge another ConnectionAttribute class into self.

        Parameters
        ----------
            ca: an instance of ConnectionAttribute class.
        """
        self.region.append(ca.region)
        self.tract.append(ca.tract)

    def remove(self):
        pass

