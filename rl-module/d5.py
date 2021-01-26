'''
MIT License
Copyright (c) Chen-Yu Yen - Soheil Abbasloo 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import threading
import logging
import tensorflow as tf
import sys
from agent import Agent
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import gym
import numpy as np
import time
import random
import datetime
import sysv_ipc
import signal
import pickle
from utils import logger, Params
from envwrapper import Env_Wrapper, TCP_Env_Wrapper, GYM_Env_Wrapper


def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)



def evaluate_TCP(env, agent, epoch, summary_writer, params, s0_rec_buffer, eval_step_counter):


    score_list = []

    eval_times = 1
    eval_length = params.dict['max_eps_steps']
    start_time = time.time()
    for _ in range(eval_times):

        step_counter = 0
        ep_r = 0.0

        if not params.dict['use_TCP']:
            s0 = env.reset()

        if params.dict['recurrent']:
            a = agent.get_action(s0_rec_buffer, False)
        else:
            a = agent.get_action(s0, False)
        a = env.map_action(a[0][0])
        # print("matthew:a:"+str(a))

        env.write_action(a,0)

        while True:

            eval_step_counter += 1
            step_counter += 1

            s1, r, terminal, error_code,ccc,rrr = env.step(a, eval_=True)

            if error_code == True:
                s1_rec_buffer = np.concatenate( (s0_rec_buffer[params.dict['state_dim']:], s1) )

                if params.dict['recurrent']:
                    a1 = agent.get_action(s1_rec_buffer, False)
                else:
                    a1 = agent.get_action(s1, False)

                a1 = env.map_action(a1[0][0])

                env.write_action(a1,0)
                print("matthew:a1:"+str(a1))

            else:
                print("Invalid state received...\n")
                env.write_action(a,0)
                continue

            ep_r = ep_r+r


            if (step_counter+1) % params.dict['tb_interval'] == 0:

                summary = tf.summary.Summary()
                summary.value.add(tag='Eval/Step/0-Actions', simple_value=env.map_action(a))
                summary.value.add(tag='Eval/Step/2-Reward', simple_value=r)
            summary_writer.add_summary(summary, eval_step_counter)

            s0 = s1
            a = a1
            if params.dict['recurrent']:
                s0_rec_buffer = s1_rec_buffer


            if step_counter == eval_length or terminal:
                score_list.append(ep_r)
                break

    summary = tf.summary.Summary()
    summary.value.add(tag='Eval/Return', simple_value=np.mean(score_list))
    summary_writer.add_summary(summary, epoch)

    return eval_step_counter



class learner_killer():

    def __init__(self, buffer):

        self.replay_buf = buffer
        print("learner register sigterm")
        signal.signal(signal.SIGTERM, self.handler_term)
        print("test length:", self.replay_buf.length_buf)
    def handler_term(self, signum, frame):
        if not config.eval:
            with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), "wb") as fp:
                pickle.dump(self.replay_buf, fp)
                print("test length:", self.replay_buf.length_buf)
                print("--------------------------Learner: Saving rp memory--------------------------")
        print("-----------------------Learner's killed---------------------")
        sys.exit(0)


class Orca_state(object):
    """
    docstring
    """
    Ordinary = 0
    EI_c_1 = 1 #
    EI_r_1 = 2 #
    EI_c_2 = 3 #receive ack of the class action and generate the reward 
    EI_r_2 = 4 #receive ack of the rl action and generate the reward

EI_sequence=0 #0:cl first 1:rl first
u_1=0
u_2=0
u_3=0

def main():

    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', default=False, help='default is  %(default)s')
    parser.add_argument('--eval', action='store_true', default=False, help='default is  %(default)s')
    parser.add_argument('--tb_interval', type=int, default=1)
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--mem_r', type=int, default = 123456)
    parser.add_argument('--mem_w', type=int, default = 12345)
    parser.add_argument('--base_path',type=str, required=True)
    parser.add_argument('--job_name', type=str, choices=['learner', 'actor'], required=True, help='Job name: either {\'learner\', actor}')
    parser.add_argument('--task', type=int, required=True, help='Task id')


    ## parameters from parser
    global config
    global params
    config = parser.parse_args()

    # print("matthew:job name:"+str(config.job_name))

    ## parameters from file
    params = Params(os.path.join(config.base_path,'params.json'))
    print("Params:"+str(params.dict))
    # print()

    #parameters of the Orca itself 
    orca_state=Orca_state.Ordinary
    th3=0.2
    


    if params.dict['single_actor_eval']:
        # print("matthew:single_actor_eval")
        local_job_device = ''
        shared_job_device = ''
        def is_actor_fn(i): return True
        global_variable_device = '/cpu'
        is_learner = False
        # print("matthew:create_local_server")
        server = tf.train.Server.create_local_server()#会在本地创建一个单进程集群，该集群众的服务默认为启动状态
        filters = []
    else:

        local_job_device = '/job:%s/task:%d' % (config.job_name, config.task)
        shared_job_device = '/job:learner/task:0'

        is_learner = config.job_name == 'learner'
        # print("matthew:is_learner1:"+str(is_learner))

        global_variable_device = shared_job_device + '/cpu'


        def is_actor_fn(i): return config.job_name == 'actor' and i == config.task


        if params.dict['remote']:
            cluster = tf.train.ClusterSpec({
                'actor': params.dict['actor_ip'][:params.dict['num_actors']],
                'learner': [params.dict['learner_ip']]
            })
        else:
            cluster = tf.train.ClusterSpec({
                    'actor': ['localhost:%d' % (8001 + i) for i in range(params.dict['num_actors'])],
                    'learner': ['localhost:8000']
                })


        server = tf.train.Server(cluster, job_name=config.job_name,
                                task_index=config.task)
        filters = [shared_job_device, local_job_device]



    if params.dict['use_TCP']:
        env_str = "TCP"
        env_peek = TCP_Env_Wrapper(env_str, params,use_normalizer=params.dict['use_normalizer'])

    else:
        env_str = 'YourEnvironment'
        env_peek =  Env_Wrapper(env_str)



    s_dim, a_dim = env_peek.get_dims_info()
    action_scale, action_range = env_peek.get_action_info()
    print("action_scale:"+str(action_scale))
    print("action_range:"+str(action_range))

    if not params.dict['use_TCP']:
        params.dict['state_dim'] = s_dim
    if params.dict['recurrent']:
        s_dim = s_dim * params.dict['rec_dim']


    if params.dict['use_hard_target'] == True:
        params.dict['tau'] = 1.0


    with tf.Graph().as_default(),\
        tf.device(local_job_device + '/cpu'):

        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        actor_op = []
        now = datetime.datetime.now()
        tfeventdir = os.path.join( config.base_path, params.dict['logdir'], config.job_name+str(config.task) )
        # print("tfeventdir:"+tfeventdir)
        params.dict['train_dir'] = tfeventdir

        if not os.path.exists(tfeventdir):
            os.makedirs(tfeventdir)
        summary_writer = tf.summary.FileWriterCache.get(tfeventdir)

        with tf.device(shared_job_device):

            agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
                        h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                        lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                        LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],noise_exp=params.dict['noise_exp'])

            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            shapes = [[s_dim], [a_dim], [1], [s_dim], [1]]
            queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name="rp_buf")


        if is_learner:
            with tf.device(params.dict['device']):
                agent.build_learn()

                agent.create_tf_summary()

            if config.load is True and config.eval==False:
                if os.path.isfile(os.path.join(params.dict['train_dir'], "replay_memory.pkl")):
                    with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), 'rb') as fp:
                        replay_memory = pickle.load(fp)

            _killsignal = learner_killer(agent.rp_buffer)


        for i in range(params.dict['num_actors']):
                if is_actor_fn(i):
                    if params.dict['use_TCP']:
                        shrmem_r = sysv_ipc.SharedMemory(config.mem_r)
                        shrmem_w = sysv_ipc.SharedMemory(config.mem_w)
                        env = TCP_Env_Wrapper(env_str, params, config=config, for_init_only=False, shrmem_r=shrmem_r, shrmem_w=shrmem_w,use_normalizer=params.dict['use_normalizer'])
                    else:
                        env = GYM_Env_Wrapper(env_str, params)

                    a_s0 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s0')
                    a_action = tf.placeholder(tf.float32, shape=[a_dim], name='a_action')
                    a_reward = tf.placeholder(tf.float32, shape=[1], name='a_reward')
                    a_s1 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s1')
                    a_terminal = tf.placeholder(tf.float32, shape=[1], name='a_terminal')
                    a_buf = [a_s0, a_action, a_reward, a_s1, a_terminal]


                    with tf.device(shared_job_device):
                        actor_op.append(queue.enqueue(a_buf))

        if is_learner:
            Dequeue_Length = params.dict['dequeue_length']
            dequeue = queue.dequeue_many(Dequeue_Length)

        queuesize_op = queue.size()

        if params.dict['ckptdir'] is not None:
            params.dict['ckptdir'] = os.path.join( config.base_path, params.dict['ckptdir'])
            print("## checkpoint dir:", params.dict['ckptdir'])
            isckpt = os.path.isfile(os.path.join(params.dict['ckptdir'], 'checkpoint') )
            print("## checkpoint exists?:", isckpt)
            if isckpt== False:
                print("\n# # # # # # Warning ! ! ! No checkpoint is loaded, use random model! ! ! # # # # # #\n")
        else:
            params.dict['ckptdir'] = tfeventdir

        tfconfig = tf.ConfigProto(allow_soft_placement=True)

        if params.dict['single_actor_eval']:
            mon_sess = tf.train.SingularMonitoredSession(
                checkpoint_dir=params.dict['ckptdir'])
        else:
            mon_sess = tf.train.MonitoredTrainingSession(master=server.target,
                    save_checkpoint_secs=15,
                    save_summaries_secs=None,
                    save_summaries_steps=None,
                    is_chief=is_learner,
                    checkpoint_dir=params.dict['ckptdir'],
                    config=tfconfig,
                    hooks=None)

        agent.assign_sess(mon_sess)


        if is_learner:

            if config.eval is True:
                print("=========================Learner is up===================")
                while not mon_sess.should_stop():
                    time.sleep(1)
                    continue

            if config.load is False:
                agent.init_target()

            counter = 0
            start = time.time()

            dequeue_thread = threading.Thread(target=learner_dequeue_thread, args=(agent,params, mon_sess, dequeue, queuesize_op, Dequeue_Length),daemon=True)
            first_time=True

            while not mon_sess.should_stop():

                if first_time == True:
                    dequeue_thread.start()
                    first_time=False

                up_del_tmp=params.dict['update_delay']/1000.0
                time.sleep(up_del_tmp)
                if agent.rp_buffer.ptr>200 or agent.rp_buffer.full :
                    agent.train_step()
                    if params.dict['use_hard_target'] == False:
                        agent.target_update()

                        if counter %params.dict['hard_target'] == 0 :
                            current_opt_step = agent.sess.run(agent.global_step)
                            logger.info("Optimize step:{}".format(current_opt_step))
                            logger.info("rp_buffer ptr:{}".format(agent.rp_buffer.ptr))

                    else:
                        if counter %params.dict['hard_target'] == 0 :

                            agent.target_update()
                            current_opt_step = agent.sess.run(agent.global_step)
                            logger.info("Optimize step:{}".format(current_opt_step))
                            logger.info("rp_buffer ptr:{}".format(agent.rp_buffer.ptr))

                counter += 1


        else:
                start = time.time()
                step_counter = np.int64(0)
                eval_step_counter = np.int64(0)
                s0 = env.reset()
                s0_rec_buffer = np.zeros([s_dim])
                s1_rec_buffer = np.zeros([s_dim])
                s0_rec_buffer[-1*params.dict['state_dim']:] = s0


                if params.dict['recurrent']:
                    a = agent.get_action(s0_rec_buffer,not config.eval)
                else:
                    a = agent.get_action(s0, not config.eval)
                a = env.map_action(a[0][0])#matthew
                env.write_action(a,orca_state)
                print("matthew:a2:"+str(a))
                prev_cwnd=a
                epoch = 0
                ep_r = 0.0
                start = time.time()
                count=0
                cwnd_list=[]
                haveWrite=False
                strCWND_record=""
                strRLCWND_record=""
                strCLCWND_record=""
                while True:
                    print("")
                    print(count)
                    count+=1
                    current_time = time.time()
                    print("matthew: time:"+str(current_time))
                    epoch += 1
                    # print("matthew:time1:"+str(time.time()))
                    # 这里的state意味着上一个episode的state，然后会根据不同情况决定下一个state是什么
                    step_counter += 1
                    s1, r, terminal, error_code,cwnd ,rtt= env.step(a,eval_=config.eval)#这个action参数没啥用
                    if(count==1):
                        prev_cwnd = cwnd 
                    
                    # print("matthew:s1:"+str(s1))
                    strCWND_record+=(str(int((current_time-start)*1000000)/1000000)+" "+str(cwnd)+"\n")

                    
                    
                    if error_code == True:
                        if orca_state==Orca_state.Ordinary:
                            print("matthew:stage:Ordinary")
                            cwnd_list.clear()
                       
                        
                            s1_rec_buffer = np.concatenate( (s0_rec_buffer[params.dict['state_dim']:], s1) )

                            if params.dict['recurrent']:
                                a_rl = agent.get_action(s1_rec_buffer, not config.eval)#a_rl是指数项
                            else:
                                a_rl = agent.get_action(s1,not config.eval)

                            print("matthew:a_rl[0][0]:"+str(a_rl[0][0]))


                            a_r= env.map_action(a_rl[0][0])#得到RL方法得到的action，得到的是prev_cwnd所要乘的一个系数 的100倍

                            x_r=prev_cwnd*a_r/100#按照rl得出的对应的cwnd的值
                            x_c=cwnd#得到cubic得到的action  它就是新cwnd的值
                            strRLCWND_record+=(str(int((current_time-start)*1000000)/1000000)+" "+str(x_r)+"\n")
                            strCLCWND_record+=(str(int((current_time-start)*1000000)/1000000)+" "+str(x_c)+"\n")
                            print("matthew:x_r:"+str(x_r)+" x_c:"+str(x_c)+" prev_cwnd:"+str(prev_cwnd))
                            if np.abs(x_r-x_c)>=0.2*prev_cwnd:#prev_cwnd not defined yet
                                #进入evaluation stage
                                if(x_r-prev_cwnd)*(x_c-prev_cwnd)>0 :#如果x_r 和x_c同增同减
                                         #进入EI,注意：进入EI和进入evaluation stage是不一样的
                                        if x_r>x_c:
                                            cwnd_list.append(x_c)
                                            cwnd_list.append(x_r)
                                            EI_sequence=1
                                            a_final=cwnd_list[0]
                                            orca_state=Orca_state.EI_c_1
                                        else:
                                            cwnd_list.append(x_r)
                                            cwnd_list.append(x_c)
                                            EI_sequence=0
                                            a_final=cwnd_list[0]
                                            orca_state=Orca_state.EI_r_1
                                else: 
                                    # if (x_c-prev_cwnd)<0 :  
                                    #     a_final=x_c
                                    #     prev_cwnd=a_final
                                    # else:
                                    #进入EI
                                    if x_r>x_c:
                                        cwnd_list.append(x_c)
                                        cwnd_list.append(x_r)
                                        EI_sequence=1
                                        a_final=cwnd_list[0]
                                        orca_state=Orca_state.EI_c_1
                                    else:
                                        cwnd_list.append(x_r)
                                        cwnd_list.append(x_c)
                                        EI_sequence=0
                                        a_final=cwnd_list[0]
                                        orca_state=Orca_state.EI_r_1    
                            else:
                                a_final=x_c
                        
                        elif orca_state==Orca_state.EI_c_1:
                            print("matthew:stage:EI_c_1")
                            if EI_sequence==1:#cl先行 下一个状态是EI_r_1
                                u_1=r/rtt#得到u1但是后续需要除episode的长度
                                print("u_1:"+str(u_1))
                                orca_state=Orca_state.EI_r_1
                                a_final=cwnd_list[1]
                            else:
                                
                                orca_state=Orca_state.EI_r_2
                                a_final=prev_cwnd
                        
                        elif orca_state==Orca_state.EI_r_1:
                            print("matthew:stage:EI_r_1")
                            if EI_sequence==0:#rl先行 下一个状态是EI_c_1
                                u_1=r/rtt#得到u1但是后续需要除episode的长度
                                print("u_1:"+str(u_1))
                                orca_state=Orca_state.EI_c_1
                                a_final=cwnd_list[1]
                            else:
                                
                                orca_state=Orca_state.EI_c_2
                                a_final=prev_cwnd

                        elif orca_state==Orca_state.EI_c_2:
                            print("matthew:stage:EI_c_2")
                            if EI_sequence==1:#cl先行 下一个动作是EI_r_2
                                u_2=2*r/rtt#得到u2但是后续需要除episode的长度
                                print("u_2:"+str(u_2))
                                orca_state=Orca_state.EI_r_2
                                # a_final=a_r
                            else:
                                orca_state=Orca_state.Ordinary#Evaluation结束进入Ordinary
                                u_3=2*r/rtt
                                print("u_1:"+str(u_1)+"  u_2:"+str(u_2)+" u_3:"+str(u_3))
                                if max(u_2,u_3)>=u_1:
                                    if(u_2>u_3):#EI——sequence=1 rl先行
                                        a_final=cwnd_list[0]
                                        prev_cwnd=a_final
                                    else:
                                        a_final=cwnd_list[1]
                                        prev_cwnd=a_final
                                else:
                                        a_final=prev_cwnd

                        elif orca_state==Orca_state.EI_r_2:
                            print("matthew:stage:EI_r_2")
                            if EI_sequence==0:#rl先行 下一个动作是EI_c_2
                                u_2=2*r/rtt#得到u2但是后续需要除episode的长度
                                orca_state=Orca_state.EI_c_2
                                # a_final=a_r
                            else:
                                orca_state=Orca_state.Ordinary#Evaluation结束进入Ordinary
                                u_3=2*r/rtt
                                print("u_1:"+str(u_1)+"  u_2:"+str(u_2)+" u_3:"+str(u_3))
                                if max(u_2,u_3)>=u_1:
                                    if(u_2>u_3):#EI——sequence=0 cl先行
                                        a_final=cwnd_list[0]
                                        prev_cwnd=a_final
                                    else:
                                        a_final=cwnd_list[1]
                                        prev_cwnd=a_final
                                else:
                                    a_final=prev_cwnd

                    # step_counter += 1
                    # s1, r, terminal, error_code,cwnd = env.step(a,eval_=config.eval)
                    # print("matthew:state:"+str(s1))

                    # if error_code == True:
                    #     s1_rec_buffer = np.concatenate( (s0_rec_buffer[params.dict['state_dim']:], s1) )

                    #     if params.dict['recurrent']:
                    #         a_rl = agent.get_action(s1_rec_buffer, not config.eval)
                    #     else:
                    #         a_rl = agent.get_action(s1,not config.eval)

                        # a_final = env.map_action(a_rl[0][0])
                        if(current_time>start+34 and haveWrite==False):
                            haveWrite=True
                            print("matthew: 30s!!!!")
                            cwnd_writer=open("/home/matthew/Orca/Analysis/cwnd.txt","w")
                            cwnd_writer.write(strCWND_record)
                            cwnd_writer.close()
                            rl_cwnd_writer=open("/home/matthew/Orca/Analysis/rl_cwnd.txt","w")
                            rl_cwnd_writer.write(strRLCWND_record)
                            rl_cwnd_writer.close()
                            cl_cwnd_writer=open("/home/matthew/Orca/Analysis/cl_cwnd.txt","w")
                            cl_cwnd_writer.write(strCLCWND_record)
                            cl_cwnd_writer.close()



                        print("matthew:time2:"+str(time.time()-start))
                        env.write_action(a_final,orca_state)
                        print("matthew:a3:"+str(a_final))

                    else:
                        print("TaskID:"+str(config.task)+"Invalid state received...\n")
                        env.write_action(a,orca_state)
                        continue

                    if params.dict['recurrent']:
                        fd = {a_s0:s0_rec_buffer, a_action:a, a_reward:np.array([r]), a_s1:s1_rec_buffer, a_terminal:np.array([terminal], np.float)}
                    else:
                        fd = {a_s0:s0, a_action:a, a_reward:np.array([r]), a_s1:s1, a_terminal:np.array([terminal], np.float)}

                    if not config.eval:
                        mon_sess.run(actor_op, feed_dict=fd)

                    s0 = s1
                    a = a_final
                    if params.dict['recurrent']:
                        s0_rec_buffer = s1_rec_buffer

                    if not params.dict['use_TCP'] and (terminal):
                        if agent.actor_noise != None:
                            agent.actor_noise.reset()

                    if (epoch% params.dict['eval_frequency'] == 0):
                        eval_step_counter = evaluate_TCP(env, agent, epoch, summary_writer, params, s0_rec_buffer, eval_step_counter)


                print("total time:", time.time()-start)

def learner_dequeue_thread(agent,params, mon_sess, dequeue, queuesize_op, Dequeue_Length):
    ct = 0
    while True:
        ct = ct + 1
        data = mon_sess.run(dequeue)
        agent.store_many_experience(data[0], data[1], data[2], data[3], data[4], Dequeue_Length)
        time.sleep(0.01)


def learner_update_thread(agent,params):
    delay=params.dict['update_delay']/1000.0
    ct = 0
    while True:
        agent.train_step()
        agent.target_update()
        time.sleep(delay)


if __name__ == "__main__":
    main()
