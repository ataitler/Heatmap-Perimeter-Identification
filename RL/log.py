class Logger(object):
    def __init__(self, file_name):
        self.file_name = file_name
        fh = open(self.file_name, 'w+')
        fh.close()

    def log_episode(self, episode, actions, rewards):
        msg = ("######################\n# Episode " + str(episode) + " \n######################\n")
        msg += "step,action,reward\n"
        for i in range(len(actions)):
            msg += str(i) + "," + str(actions[i]) + "," + str(rewards[i]) + "\n"

        fh = open(self.file_name, 'a')
        fh.write(msg)
        fh.close()
