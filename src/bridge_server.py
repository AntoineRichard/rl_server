#!/usr/bin/env python

import rospy
from bottle import Bottle, request
import json
import numpy as np
from std_msgs.msg import Bool 
from rl_server.msg import Episode
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from heron_msgs.msg import Drive


class Server:
    def __init__(self):
        # ROS
        self.episode_manager_pub_ = rospy.Publisher('/server/episode_manager', Episode, queue_size=1)  
        self.action_pub_ = rospy.Publisher('/cmd_rl', Drive, queue_size=1)
        self.sim_ok_pub_ = rospy.Publisher('/server/sim_ok', Bool, queue_size=1)  
        rospy.Subscriber('/agent/is_done', Bool, self.doneCallback)
        self.service_name_ = rospy.get_param('~service_name','')
        self.path_to_data_ = rospy.get_param('~path_to_pose','blabla')
        self.drive = Drive()
        self.drive.left=0
        self.drive.right=0
        # SERVICE
        self.msg_ = ModelState()
        self.msg_.model_name='heron'
        self.msg_.pose.position.x = 0
        self.msg_.pose.position.y = 0
        self.msg_.pose.position.z = 0.145
        self.msg_.pose.orientation.x = 0
        self.msg_.pose.orientation.y = 0
        self.msg_.pose.orientation.z = 0
        self.msg_.pose.orientation.w = 0
        self.msg_.twist.linear.x = 0
        self.msg_.twist.linear.y = 0
        self.msg_.twist.linear.z = 0
        self.msg_.twist.angular.x = 0
        self.msg_.twist.angular.y = 0
        self.msg_.twist.angular.z = 0

        # SERVER
        self._host = 'localhost'
        self._port = 8080
        self._app = Bottle()
        self._route()
        # VARS
        self.expected_keys = ['random', 'steps', 'repeat', 'discount', 'training', 'current_step', 'reward']
        self.check_rate_ = 2.0
        self.op_OK_ = False
        self.poses = np.load(self.path_to_data_)[1000:]

    def _route(self):
        self._app.route('/toServer', method="POST", callback=self._onPost)
 
    def start(self):
        self._app.run(host=self._host, port=self._port, reloarder=False)

    def doneCallback(self, msg):
        if msg.data:
            self.op_OK_=True

    def _onPost(self):
        req = json.loads(request.body.read())
        ep = Episode()
        for i in req.keys():
            if i  not in self.expected_keys:
                raise ValueError('incorrect post request. You must provide the following fields: \"steps\", \"random\", and \"repeat\"')
        ep.steps = req['steps']
        ep.random_agent = (req['random'] == 1)
        ep.discount = req['discount']
        ep.training = (req['training'] == 1)
        rospy.wait_for_service(self.service_name_)
        for i in range(req['repeat']+1):
            self.op_OK_ = False
            try:
                set_state = rospy.ServiceProxy(self.service_name_, SetModelState)
                idx = int(np.random.rand(1)*self.poses.shape[0])
                self.msg_.pose.position.x = self.poses[idx][0]
                self.msg_.pose.position.y = self.poses[idx][1]
                self.msg_.pose.orientation.x = self.poses[idx][4]
                self.msg_.pose.orientation.y = self.poses[idx][5]
                self.msg_.pose.orientation.z = self.poses[idx][6]
                self.msg_.pose.orientation.w = self.poses[idx][7]
         
                resp = set_state(self.msg_)
                print("refresh Ok new boat pose: x:",self.poses[idx][0]," y:",self.poses[idx][1])
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
            rospy.sleep(1.0)
            self.episode_manager_pub_.publish(ep)
            rospy.sleep(1.0)
            self.sim_ok_pub_.publish(True)
            while ((not self.op_OK_) and (not rospy.is_shutdown())):
                #print('sleeping')
                rospy.sleep(self.check_rate_)
            self.action_pub_.publish(self.drive)
            try:
                set_state = rospy.ServiceProxy(self.service_name_, SetModelState)
                resp = set_state(self.msg_)
                print("0 velocity refresh")
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
        return 'Done'

if __name__ == "__main__":
    rospy.init_node('server')
    server = Server()
    server.start()
