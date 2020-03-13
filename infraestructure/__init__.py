class InfraestructureDevice(object):

    def __init__(self, node_id, incoming_links, children):
        self.__id = node_id
        self.__incoming_links = incoming_links
        self.__children = children
        self.__father = None


    def update_link_info(self, link_id):
        raise NotImplementedError

    def get_net_info(self):
        raise NotImplementedError

    def update_father(self, father):
        raise NotImplementedError
