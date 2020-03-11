'''
Created on 25/11/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
from xml.dom import minidom
import datetime

# convert SUMO cfg/net files to the graph format accepted by the KSP script
def convert_SUMO_to_KSP(file_name):
    
    net_file_name = file_name
    
    # check if the argument corresponds to a net or cfg file
    if len(minidom.parse(file_name).getElementsByTagName('net')) == 0:
        net_file_name = minidom.parse(file_name).getElementsByTagName('net-file')[0].attributes['value'].value
        if '/' in file_name:
            net_file_name = '%s/%s' % (file_name[:file_name.rfind('/')], net_file_name)
    
    # print a description header
    print '##################################################'
    print '#                                                #'
    print '# Network file automatically generated in        #'
    print '# compliance with the KSP.py script.             #'
    print '#                                                #'
    print '# WARNING! This file is compatible with the      #'
    print '# Maslab\'s network file specification but is not #'
    print '# complete. It is intended for the specific      #'
    print '# purpose of the present framework.              #'
    print '#                                                #'
    print '# Generated on %s                       #' % datetime.date.today().strftime('%2d-%b-%Y')
    print '#                                                #'
    print '##################################################'
    
    # print the function type
    print 'function simple_function (f) fftt+f-f'
    
    # process the nodes
    nodes = minidom.parse(net_file_name).getElementsByTagName('junction')
    for n in nodes:
        # internal nodes are not considered
        if n.attributes['type'].value != 'internal':
            print 'node %s' % (n.attributes['id'].value)
    
    # process the edges
    edges = minidom.parse(net_file_name).getElementsByTagName('edge')
    for e in edges:
        # only "normal" edges are considered (those that do not have the "function" attribute are not considered either)
        if 'function' not in e.attributes.items()[0] or e.attributes['function'].value == 'normal':
            
            # get the edges' length (it can be extracted from the inner lane tag or from the edge tag itself)
            if 'length' not in e.attributes.items()[0]:
                length = e.getElementsByTagName('lane')[0].attributes['length'].value
            else:
                length = e.attributes['length'].value
            
            # get the edges' maximum speed (it can be extracted from the inner lane tag or from the edge tag itself)
            if 'speed' not in e.attributes.items()[0]:
                speed = e.getElementsByTagName('lane')[0].attributes['speed'].value
            else:
                speed = e.attributes['speed'].value
            
            # calculate the weight/cost of the edge
            weight = (float(length) / float(speed)) / 60
            
            # print as an arc
            print 'dedge %s %s %s simple_function %f' % (e.attributes['id'].value, e.attributes['from'].value, e.attributes['to'].value, weight)
    
    # OD pairs are not printed here because they are no used
    