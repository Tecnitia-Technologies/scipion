from Tkinter import *

'''
Taken from following forum:
http://code.activestate.com/recipes/52266-multilistbox-tkinter-widget/
This is a compound widget that gangs multiple Tk Listboxes to a single scrollbar to achieve a simple 
multi-column scrolled listbox. Most of the Listbox API is mirrored to make it act like the normal 
Listbox but with multiple values per row.
'''
class MultiListbox(PanedWindow):
    """MultiListbox class for creating a Grid widget"""
    def __init__(self,master,lists):
        PanedWindow.__init__(self,master,borderwidth=1,showhandle=False,sashwidth=2,sashpad=0,relief=SUNKEN)
        self.lists = []
        self.columns=[]
        for l, w in lists:
            self.columns.append(l)
            frame = Frame(self); frame.pack(side=LEFT, expand=YES, fill=BOTH)
            tl=Label(frame, text=l, borderwidth=2, relief=GROOVE)
            tl.pack(fill=X)
            tl.bind('<Button-1>',self.clickon)
            lb = Listbox(frame, width=w, borderwidth=0, selectborderwidth=0,relief=FLAT, exportselection=FALSE, selectmode=MULTIPLE ,bg='white')
            lb.pack(expand=YES, fill=BOTH)
            self.lists.append(lb)
            lb.bind('<B1-Motion>', lambda e, s=self: s._select(e.y))
            lb.bind('<Button-1>', lambda e, s=self: s._select(e.y))
            lb.bind('<Double-Button-1>', self._doubleClick)
            lb.bind('<Leave>', lambda e: 'break')
            lb.bind('<B2-Motion>', lambda e, s=self: s._b2motion(e.x, e.y))
            lb.bind('<Button-2>', lambda e, s=self: s._button2(e.x, e.y))
            lb.bind('<Button-3>', lambda e, s=self: s._button3(e.x, e.y))
            lb.bind('<Button-4>', lambda e, s=self: s._scroll(SCROLL, 1, PAGES))
            lb.bind('<Button-5>', lambda e, s=self: s._scroll(SCROLL, -1, PAGES))
            self.add(frame)
        Label(master, borderwidth=1, relief=FLAT).pack(fill=X)
        sb = Scrollbar(master, orient=VERTICAL, command=self._scroll,borderwidth=1)
        sb.pack(fill=Y,side=RIGHT,expand=NO)
        for l in self.lists:
            l['yscrollcommand']=sb.set
        self.add(frame)
        self.pack(expand=YES,fill=BOTH, side=TOP)
        self.sortedBy = -1
        self.previousWheel = 0
        self.SelectCallback = None
        self.DoubleClickCallback = None
        self.AllowSort = True

    def _doubleClick(self, event=''):
        if self.DoubleClickCallback:
            self.DoubleClickCallback()

    def _select(self, y,state=16):
        row = self.lists[0].nearest(y)
        if state==16:self.selection_clear(0, END)
        self.selection_set(row)
        return 'break'

    def _button2(self, x, y):
        for l in self.lists: l.scan_mark(x, y)
        return 'break'

    def _button3(self, x, y):
        print "right click"
        #self._select(y, state)
        return 'break'
    

    def _b2motion(self, x, y):
        for l in self.lists: l.scan_dragto(x, y)        
        return 'break'


    def _scroll(self, *args):
        for l in self.lists:
            apply(l.yview, args)
        return 'break'


    def clickon(self,e):
        if self.AllowSort:
            self._sortBy(self.columns.index(e.widget['text']))


    def _sortBy(self, column):
        """ Sort by a given column. """
        if column == self.sortedBy:
            direction = -1 * self.direction
        else:
            direction = 1

        elements = self.get(0, END)
        self.delete(0, END)
        elements.sort(lambda x, y: self._sortAssist(column, direction, x, y))
        self.insert(END, *elements)

        self.sortedBy = column
        self.direction = direction


    def _sortAssist(self, column, direction, x, y):
        c = cmp(x[column], y[column])
        if c:
            return direction * c
        else:
            return direction * cmp(x, y)

    def curselection(self):
        return self.lists[0].curselection()


    def delete(self, first, last=None):
        for l in self.lists:
            l.delete(first, last)


    def get(self, first, last=None):
        result = []
        for l in self.lists:
            result.append(l.get(first,last))
        if last: return apply(map, [None] + result)
        return result


    def index(self, index):
        self.lists[0].index(index)


    def insert(self, index, *elements):
        for e in elements:
            i = 0
            for l in self.lists:
                l.insert(index, e[i])
                i = i + 1

    def size(self):
        return self.lists[0].size()

    def see(self, index):
        for l in self.lists:
            l.see(index)

    def selection_anchor(self, index):
        for l in self.lists:
            l.selection_anchor(index)

    def selection_clear(self, first, last=None):
        for l in self.lists:
            l.selection_clear(first, last)


    def selection_includes(self, index):
        return self.lists[0].selection_includes(index)


    def selectedIndex(self):
        if len(self.curselection()) > 0:
            return int(self.curselection()[0])
        return -1
    
    def selection_set(self, first, last=None):
        for l in self.lists:
            l.selection_set(first, last)
        #print self.curselection()
        index = self.selectedIndex()
        if self.SelectCallback:
            self.SelectCallback(index)

def getGeometry(win):
    return win.winfo_reqwidth(), win.winfo_reqheight(), win.winfo_x(), win.winfo_y()

def centerWindows(root):
    """Center a windows in the middle of the screen"""
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    gw, gh, gx, gy = getGeometry(root)
    x = w / 2 - gw / 2
    y = h / 2 - gh / 2
    root.geometry("%dx%d+%d+%d" % (gw, gh, x, y))

__tipwindow = None

# Creates a tooptip box for a widget.
def createToolTip( widget, text ):
    def enter( event ):
        global __tipwindow
        x = y = 0
        if __tipwindow or not text:
            return
        x, y, cx, cy = widget.bbox( "insert" )
        x += widget.winfo_rootx() + 27
        y += widget.winfo_rooty() + 27
        # Creates a toplevel window
        __tipwindow = tw = Toplevel( widget )
        # Leaves only the label and removes the app window
        tw.wm_overrideredirect( 1 )
        tw.wm_geometry( "+%d+%d" % ( x, y ) )
        label = Label( tw, text = text, justify = LEFT,
                       background = "#ffffe0", relief = SOLID, borderwidth = 1,
                       font = ( "tahoma", "8", "normal" ) )
        label.pack( ipadx = 1 )
        
    def close( event ):
        global __tipwindow
        tw = __tipwindow
        __tipwindow = None
        if tw:
            tw.destroy()
            
    widget.bind( "<Enter>", enter )
    widget.bind( "<Leave>", close )


'''Michael Lange <klappnase (at) freakmail (dot) de>
The ToolTip class provides a flexible tooltip widget for Tkinter; it is based on IDLE's ToolTip
module which unfortunately seems to be broken (at least the version I saw).
INITIALIZATION OPTIONS:
anchor :        where the text should be positioned inside the widget, must be on of "n", "s", "e", "w", "nw" and so on;
                default is "center"
bd :            borderwidth of the widget; default is 1 (NOTE: don't use "borderwidth" here)
bg :            background color to use for the widget; default is "lightyellow" (NOTE: don't use "background")
delay :         time in ms that it takes for the widget to appear on the screen when the mouse pointer has
                entered the parent widget; default is 1500
fg :            foreground (i.e. text) color to use; default is "black" (NOTE: don't use "foreground")
follow_mouse :  if set to 1 the tooltip will follow the mouse pointer instead of being displayed
                outside of the parent widget; this may be useful if you want to use tooltips for
                large widgets like listboxes or canvases; default is 0
font :          font to use for the widget; default is system specific
justify :       how multiple lines of text will be aligned, must be "left", "right" or "center"; default is "left"
padx :          extra space added to the left and right within the widget; default is 4
pady :          extra space above and below the text; default is 2
relief :        one of "flat", "ridge", "groove", "raised", "sunken" or "solid"; default is "solid"
state :         must be "normal" or "disabled"; if set to "disabled" the tooltip will not appear; default is "normal"
text :          the text that is displayed inside the widget
textvariable :  if set to an instance of Tkinter.StringVar() the variable's value will be used as text for the widget
width :         width of the widget; the default is 0, which means that "wraplength" will be used to limit the widgets width
wraplength :    limits the number of characters in each line; default is 150

WIDGET METHODS:
configure(**opts) : change one or more of the widget's options as described above; the changes will take effect the
                    next time the tooltip shows up; NOTE: follow_mouse cannot be changed after widget initialization

Other widget methods that might be useful if you want to subclass ToolTip:
enter() :           callback when the mouse pointer enters the parent widget
leave() :           called when the mouse pointer leaves the parent widget
motion() :          is called when the mouse pointer moves inside the parent widget if follow_mouse is set to 1 and the
                    tooltip has shown up to continually update the coordinates of the tooltip window
coords() :          calculates the screen coordinates of the tooltip window
create_contents() : creates the contents of the tooltip window (by default a Tkinter.Label)
'''
# Ideas gleaned from PySol

class ToolTip:
    def __init__(self, master, text='Your text here', delay=1500, **opts):
        self.master = master
        self._opts = {'anchor':'center', 'bd':1, 'bg':'lightyellow', 'delay':delay, 'fg':'black',\
                      'follow_mouse':0, 'font':None, 'justify':'left', 'padx':4, 'pady':2,\
                      'relief':'solid', 'state':'normal', 'text':text, 'textvariable':None,\
                      'width':0, 'wraplength':150}
        self.configure(**opts)
        self._tipwindow = None
        self._id = None
        self._id1 = self.master.bind("<Enter>", self.enter, '+')
        self._id2 = self.master.bind("<Leave>", self.leave, '+')
        self._id3 = self.master.bind("<ButtonPress>", self.leave, '+')
        self._follow_mouse = 0
        if self._opts['follow_mouse']:
            self._id4 = self.master.bind("<Motion>", self.motion, '+')
            self._follow_mouse = 1
    
    def configure(self, **opts):
        for key in opts:
            if self._opts.has_key(key):
                self._opts[key] = opts[key]
            else:
                KeyError = 'KeyError: Unknown option: "%s"' %key
                raise KeyError
    
    ##----these methods handle the callbacks on "<Enter>", "<Leave>" and "<Motion>"---------------##
    ##----events on the parent widget; override them if you want to change the widget's behavior--##
    
    def enter(self, event=None):
        self._schedule()
        
    def leave(self, event=None):
        self._unschedule()
        self._hide()
    
    def motion(self, event=None):
        if self._tipwindow and self._follow_mouse:
            x, y = self.coords()
            self._tipwindow.wm_geometry("+%d+%d" % (x, y))
    
    ##------the methods that do the work:---------------------------------------------------------##
    
    def _schedule(self):
        self._unschedule()
        if self._opts['state'] == 'disabled':
            return
        self._id = self.master.after(self._opts['delay'], self._show)

    def _unschedule(self):
        id = self._id
        self._id = None
        if id:
            self.master.after_cancel(id)

    def _show(self):
        if self._opts['state'] == 'disabled':
            self._unschedule()
            return
        if not self._tipwindow:
            self._tipwindow = tw = Toplevel(self.master)
            # hide the window until we know the geometry
            tw.withdraw()
            tw.wm_overrideredirect(1)

            if tw.tk.call("tk", "windowingsystem") == 'aqua':
                tw.tk.call("::tk::unsupported::MacWindowStyle", "style", tw._w, "help", "none")

            self.create_contents()
            tw.update_idletasks()
            x, y = self.coords()
            tw.wm_geometry("+%d+%d" % (x, y))
            tw.deiconify()
    
    def _hide(self):
        tw = self._tipwindow
        self._tipwindow = None
        if tw:
            tw.destroy()
                
    ##----these methods might be overridden in derived classes:----------------------------------##
    
    def coords(self):
        # The tip window must be completely outside the master widget;
        # otherwise when the mouse enters the tip window we get
        # a leave event and it disappears, and then we get an enter
        # event and it reappears, and so on forever :-(
        # or we take care that the mouse pointer is always outside the tipwindow :-)
        tw = self._tipwindow
        twx, twy = tw.winfo_reqwidth(), tw.winfo_reqheight()
        w, h = tw.winfo_screenwidth(), tw.winfo_screenheight()
        # calculate the y coordinate:
        if self._follow_mouse:
            y = tw.winfo_pointery() + 20
            # make sure the tipwindow is never outside the screen:
            if y + twy > h:
                y = y - twy - 30
        else:
            y = self.master.winfo_rooty() + self.master.winfo_height() + 3
            if y + twy > h:
                y = self.master.winfo_rooty() - twy - 3
        # we can use the same x coord in both cases:
        x = tw.winfo_pointerx() - twx / 2
        if x < 0:
            x = 0
        elif x + twx > w:
            x = w - twx
        return x, y

    def create_contents(self):
        opts = self._opts.copy()
        for opt in ('delay', 'follow_mouse', 'state'):
            del opts[opt]
        label = Label(self._tipwindow, **opts)
        label.pack()

##---------demo code-----------------------------------##

from tkSimpleDialog import Dialog
'''Implement a Listbox Dialog, it will return
the index selected in the lisbox or -1 on Cancel'''
class ListboxDialog(Dialog):
    def __init__(self, master, itemList, **kargs):
        self.list = itemList
        self.kargs = kargs
        Dialog.__init__(self, master)        
        
    def body(self, master):
        self.result = []
        self.lb = Listbox(master, selectmode=EXTENDED, bg="white")
        self.lb.config(**self.kargs)
        self.lb.pack(fill=BOTH)
        self.lb.bind('<Double-Button-1>', self.ok)
        for item in self.list:
            self.lb.insert(END, item)
        if len(self.list) > 0:
            self.lb.selection_set(0)
        return self.lb # initial focus

    def buttonbox(self):
        box = Frame(self)
        w = Button(box, text="OK", width=7, command=self.ok, default=ACTIVE)
        w.pack(side=RIGHT, padx=5, pady=5)
        self.bind("<Return>", self.ok)
        box.pack()
        
    def apply(self):
        self.result = map(int, self.lb.curselection())

def demo():
    root = Tk(className='ToolTip-demo')
    l = Listbox(root)
    l.insert('end', "I'm a listbox")
    l.pack(side='top')
    t1 = ToolTip(l, follow_mouse=1, text="I'm a tooltip with follow_mouse set to 1, so I won't be placed outside my parent")
    b = Button(root, text='Quit', command=root.quit)
    b.pack(side='bottom')
    t2 = ToolTip(b, text='Enough of this')
    root.mainloop()

if __name__ == '__main__':
    demo() 

