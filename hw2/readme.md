Your submission PR needs to create a sub-directory using exactly your GitHub
username as your working directory (``$WORK``).  The PR needs to be created
against the ``hw2`` branch.  (Not ``master``!)

Question 1
==========

Reimplement the class ``Line`` by using STL containers instead of raw pointers:

.. code-block:: cpp

  class Line
  {
  public:
      Line();
      Line(Line const & );
      Line(Line       &&);
      Line & operator=(Line const & );
      Line & operator=(Line       &&);
      Line(size_t size);
      ~Line();
      size_t size() const;
      float & x(size_t it) const;
      float & x(size_t it);
      float & y(size_t it) const;
      float & y(size_t it);
  private:
      // Member data.
  }; /* end class Line */

  int main(int, char **)
  {
      Line line(3);
      line.x(0) = 0; line.y(0) = 1;
      line.x(1) = 1; line.y(1) = 3;
      line.x(2) = 2; line.y(2) = 5;

      Line line2(line);
      line2.x(0) = 9;

      std::cout << "line: number of points = " << line.size() << std::endl;
      for (size_t it=0; it<line.size(); ++it)
      {
          std::cout << "point " << it << ":"
                    << " x = " << line.x(it)
                    << " y = " << line.y(it) << std::endl;
      }

      std::cout << "line2: number of points = " << line.size() << std::endl;
      for (size_t it=0; it<line.size(); ++it)
      {
          std::cout << "point " << it << ":"
                    << " x = " << line2.x(it)
                    << " y = " << line2.y(it) << std::endl;
      }

      return 0;
  }

You may copy-n-paste the above ``main`` function.  It's not suggested to
copy-n-paste the class declaration.

In a sub-directory under your ``$WORK`` directory, the code needs to be
built by executing ``make``.  When a source file changes, ``make`` needs to
pick it up and rebuild.  ``make run`` needs to produce the correct terminal
output, without crashing.  ``make clean`` needs to remove all the built and
intermediate files.

Question 2
==========

Write a C++ function that calculates the angle (in radians) between two vectors
in the 2-dimensional Cartesian coordinate system.  Use pybind11 to wrap it to
Python.  Use Python unit-test to check the result is correct.  You may use
third-party test runners, e.g., py.test or nosetest.

Similar to Question 1, in a sub-directory under your ``$WORK`` directory,
running ``make`` should build your code.  ``make test`` runs the Python-based
tests.  ``make clean`` removes all the built and intermediate files.
