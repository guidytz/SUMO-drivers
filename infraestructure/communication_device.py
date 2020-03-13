from infraestructure import InfraestructureDevice

class CommunicationDevice(InfraestructureDevice):

    def __init__(self, node_id, incoming_links, children):
        super(CommunicationDevice, self).__init__(node_id, incoming_links, children)

        for child in self.__children:
            child.update_father(self)

    def update_link_info(self, link_id):
        raise NotImplementedError

    def get_net_info(self):
        raise NotImplementedError

    def update_father(self, father):
        self.__father = father
