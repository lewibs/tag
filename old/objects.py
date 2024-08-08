import graphics

class Object:
    def __init__(self, id):
        self.id = id
        self.needs_update = True
        self.object = None

    def _remake_object(self):
        pass

    def draw(self, win):
        if self.needs_update:
            self.needs_update = False
            if self.object:
                self.undraw()
            self._remake_object()
            self.object.draw(win)

    def undraw(self):
        self.object.undraw()
        

class Point(Object):
    def __init__(self, id, color, x, y):
        super().__init__(id)
        self.set_color(color)
        self.set_position(x,y)
        self.object = graphics.Point(x, y)
        self.object.setFill(color)

    def _remake_object(self):
        self.object = graphics.Point(self.x, self.y)
        self.object.setFill(self.color)

    def set_color(self, color):
        self.needs_update = True
        self.color = color

    def set_position(self, x,y):
        self.x = int(x)
        self.y = int(y)
        self.needs_update = True