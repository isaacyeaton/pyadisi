from atom.api import Atom, Unicode, Range, Bool, Value, Int, Tuple, observe
import enaml
from enaml.qt.qt_application import QtApplication


class Calibration(Atom):
    """ A simple class representing a person object.

    """

    fig1, ax1 = plt.subplots()
    p1 = ax1.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    cursor = widgets.Cursor(ax1, useblit=True, color='red', linewidth=1)

    fig2, ax2 = plt.subplots()
    ax2.plot(np.random.normal(0, .5, 300), np.random.normal(1, .3, 300))

    debug = Bool(False)

    @observe('age')
    def debug_print(self, change):
        """ Prints out a debug message whenever the person's age changes.

        """
        if self.debug:
            templ = "{first} {last} is {age} years old."
            s = templ.format(
                first=self.first_name, last=self.last_name, age=self.age,
            )
            print s


class Employer(Person):
    """ An employer is a person who runs a company.

    """
    # The name of the company
    company_name = Unicode()


class Employee(Person):
    """ An employee is person with a boss and a phone number.

    """
    # The employee's boss
    boss = Value(Employer)

    # The employee's phone number as a tuple of 3 ints
    phone = Tuple(Int())

    # This method will be called automatically by atom when the
    # employee's phone number changes
    def _phone_changed(self, val):
        print 'received new phone number for %s: %s' % (self.first_name, val)


if __name__ == '__main__':

    img1 = plt.imread('bat_study/calibration/cam1.tif')
    cal = Calibration(data=img1, npts=60)

    with enaml.imports():
        from calibration_view import CalibrationView

    app = QtApplication()
    view = CalibrationView(calibration=cal)
    view.show()

    app.start()


if __name__ == '__main__':
    # Create an employee with a boss
    boss_john = Employer(
        first_name='John', last_name='Paw', company_name="Packrat's Cats",
    )
    employee_mary = Employee(
        first_name='Mary', last_name='Sue', boss=boss_john,
        phone=(555, 555, 5555),
    )

    # Import our Enaml EmployeeView
    with enaml.imports():
        from employee_view import EmployeeView

    app = QtApplication()
    # Create a view and show it.
    view = EmployeeView(employee=employee_mary)
    view.show()

    app.start()
