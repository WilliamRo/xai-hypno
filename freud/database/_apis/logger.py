from roma import console



class Logger(object):

  prompt = 'Logger'

  def show_status(self, text): console.show_status(text, prompt=self.prompt)
