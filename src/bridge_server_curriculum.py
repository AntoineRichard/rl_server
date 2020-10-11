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
from uuv_world_ros_plugins_msgs.srv import SetCurrentVelocity
from uuv_gazebo_ros_plugins_msgs.srv import SetFloat
import sampling_helper as sh

class Server:
    def __init__(self):
        # ROS
        self.episode_manager_pub_ = rospy.Publisher('/server/episode_manager', Episode, queue_size=1)  
        self.action_pub_ = rospy.Publisher('/cmd_rl', Drive, queue_size=1)
        self.sim_ok_pub_ = rospy.Publisher('/server/sim_ok', Bool, queue_size=1)  
        rospy.Subscriber('/agent/is_done', Bool, self.doneCallback)
        self.spawn_service_ = rospy.get_param('~spawn_service','')
        self.current_service_ = rospy.get_param('~current_service','')
        self.damping_service_ = rospy.get_param('~damping_service','')
        #self.density_service_ = rospy.get_param('~density_service','')
        #self.weight_service_ = rospy.get_param('~weight_service','')
        self.path_to_data_ = rospy.get_param('~path_to_pose','blabla')
        self.drive = Drive()
        self.drive.left=0
        self.drive.right=0
        self.current_srv = SetCurrentVelocity()
        self.damping_srv = SetFloat()
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
        self.expected_keys = ['random', 'steps', 'repeat', 'discount', 'training']
        self.check_rate_ = 2.0
        self.op_OK_ = False
        #self.poses = np.load(self.path_to_data_)[1000:]
        self.nav_line_pose, self.running_curvature, self.spawn_poses, self.map_dist2optimal_nav = sh.load_and_process(self.path_to_data_)
        self.p = (0.5*np.arange(101)/101+0.5)
        self.p = self.p/np.sum(self.p)
        self.p1 = (3 - 2.5*np.arange(101)/101)
        self.p1 = self.p1/np.sum(self.p1)
        self.offset = [-1000,-3000]
        self.current = np.linspace(0., 0.4, 20)
        self.damping = np.linspace(1.5, 3., 20)

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
        rospy.wait_for_service(self.spawn_service_)
        for i in range(req['repeat']+1):
            self.op_OK_ = False
            try:
                set_state = rospy.ServiceProxy(self.spawn_service_, SetModelState)
                pose = sh.sample_boat_position(self.nav_line_pose, self.offset,
                                               curvature = self.running_curvature, p_curvature = self.p1,
                                               p_hardspawn = self.p, hard_spawn_poses = self.spawn_poses,
                                               hard_spawn_cost = self.map_dist2optimal_nav)
                self.msg_.pose.position.x = pose[0]
                self.msg_.pose.position.y = pose[1]
                self.msg_.pose.orientation.x = pose[3]
                self.msg_.pose.orientation.y = pose[4]
                self.msg_.pose.orientation.z = pose[5]
                self.msg_.pose.orientation.w = pose[6]
         
                resp = set_state(self.msg_)
                print("refresh Ok new boat pose: x:",pose[0]," y:",pose[1])
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
            rospy.sleep(1.0)
            self.episode_manager_pub_.publish(ep)
            rospy.sleep(1.0)
            damping = np.random.choice(self.damping, replace=True)
            velocity = np.random.choice(self.current, replace=True)
            horizontal_angle = np.random.rand(1)[0]*2*np.pi
            try:
                set_current = rospy.ServiceProxy(self.current_service_, SetCurrentVelocity)
                resp = set_current(velocity, horizontal_angle, 0)
                print("refresh with current velocity of:"+str(velocity)+" with angle:"+str(horizontal_angle))
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
            try:
                set_damping = rospy.ServiceProxy(self.damping_service_, SetFloat)
                resp = set_damping(damping)
                print("refresh with damping scale of: "+str(damping))
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
            #try:
            #    set_weight = rospy.ServiceProxy(self.weight_service_, self.weight)
            #except rospy.ServiceException, e:
            #    print "Service call failed: %s" % e
            #try:
            #    set_density = rospy.ServiceProxy(self.density_service_, self.density)
            #except rospy.ServiceException, e:
            #    print "Service call failed: %s" % e

            self.sim_ok_pub_.publish(True)
            while ((not self.op_OK_) and (not rospy.is_shutdown())):
                #print('sleeping')
                rospy.sleep(self.check_rate_)
            self.action_pub_.publish(self.drive)
            try:
                set_current = rospy.ServiceProxy(self.current_service_, SetCurrentVelocity)
                resp = set_current(0, 0, 0)
                print("refresh with current velocity of: 0 with angle: 0")
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
            try:
                set_state = rospy.ServiceProxy(self.spawn_service_, SetModelState)
                resp = set_state(self.msg_)
                print("0 velocity refresh")
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
            try:
                set_current = rospy.ServiceProxy(self.current_service_, SetCurrentVelocity)
                resp = set_current(0, 0, 0)
                print("refresh with current velocity of: 0 with angle: 0")
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
        return 'Done'

if __name__ == "__main__":
    rospy.init_node('server')
    server = Server()
    server.start()
