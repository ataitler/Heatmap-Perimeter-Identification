class Logger(object):
    def __init__(self, file_name):
        self.file_name = file_name
        fh = open(self.file_name, 'w+')
        fh.close()

    def log_episode(self, episode, actions, rewards):
        str = ("######################\n"
               "# Episode " + str(episode) + " \n"
                                             "######################\n")
        str += "step,action,reward\n"
        for i in range(len(actions)):
            str += str(i) + "," + str(actions[i]) + "," + str(rewards[i]) + "\n"

        fh = open(self.file_name, 'a')
        fh.write(str)
        fh.close()
