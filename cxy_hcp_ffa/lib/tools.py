import numpy as np


def bfs(edge_list, start, end, deep_limit=np.inf):
    """
    Return a one of the shortest paths between start and end in a graph.
    The shortest path means a route that goes through the fewest vertices.
    There may be more than one shortest path between start and end.
    But the function just return one of them according to the first find.
    The function takes advantage of the Breadth First Search.
    Parameters
    ----------
    edge_list : dict | list
        The indices are vertices of a graph.
        One index's corresponding element is a collection of vertices which connect with the index.
    start : integer
        path's start vertex's id
    end : integer
        path's end vertex's id
    deep_limit : integer
        Limit the search depth to keep off too much computation.
        The deepest depth is specified by deep_limit.
        If the search depth reach the limitation without finding the end vertex, it returns False.
    Returns
    -------
    List
        one of the shortest paths
        If the list is empty, it means we can't find a path between
        the start and end vertices within the limit of deep_limit.
    """

    if start == end:
        return [start]

    tmp_path = [start]
    path_queue = [tmp_path]  # a queue used to load temporal paths
    old_nodes = [start]

    while path_queue:

        tmp_path = path_queue.pop(0)
        if len(tmp_path) > deep_limit:
            return []
        last_node = tmp_path[-1]

        for link_node in edge_list[last_node]:

            # avoid repetitive detection for a node
            if link_node in old_nodes:
                continue
            else:
                old_nodes.append(link_node)

            if link_node == end:
                # find one of the shortest path
                return tmp_path + [link_node]
            elif link_node not in tmp_path:
                # ready for deeper search
                path_queue.append(tmp_path + [link_node])

    return []
