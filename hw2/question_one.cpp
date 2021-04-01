#include <vector>
#include <iostream>

class Line
{
	public:
		Line()
		{
			std::cout << "Line is being created." << std::endl;
		}
		Line(const Line &obj)
		{
			x_member = obj.x_member;
			y_member = obj.y_member;
		}
		Line(size_t size)
		{
			for(size_t t=0; t<size; t++)
			{
				x_member.push_back(0);
				y_member.push_back(0);
			}
		}
		~Line()
		{
			std::cout << "Line is being destoryed." << std::endl;
		}
		size_t size() const
		{
			return (size_t)x_member.size();
		}
		void x(int t, int value)
		{
			x_member[t] = value;
		}
		void y(int t, int value)
		{
			y_member[t] = value;
		}
		int x(size_t t) const
		{
			return x_member[t];
		}
		int y(size_t t) const
		{
			return y_member[t];
		}
	private:
		std::vector <int> x_member;
		std::vector <int> y_member;
};

int main()
{
    Line line(3);
    line.x(0,0); line.y(0,1);
    line.x(1,1); line.y(1,3);
    line.x(2,2); line.y(2,5);

    Line line2(line);
    line2.x(0,9);

    std::cout << "line: number of points = " << line.size() << std::endl;
    for (size_t it=0; it<line.size(); it++)
    {
        std::cout << "point " << it << ":"
                  << " x = " << line.x(it)
                  << " y = " << line.y(it) << std::endl;
    }

    std::cout << "line2: number of points = " << line2.size() << std::endl;
    for (size_t it=0; it<line2.size(); it++)
    {
        std::cout << "point " << it << ":"
                  << " x = " << line2.x(it)
                  << " y = " << line2.y(it) << std::endl;
    }

    return 0;
}