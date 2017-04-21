// The MIT License (MIT)
//
// Copyright (c) 2017 Miguel Ascanio Gómez mascanio13@gmail.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <cmath>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
using namespace std;

class TicToc {

private:

public:
	bool active;
	timespec start, end;

	time_t sec;
	long nsec;

	void diff();

	TicToc(bool active = true) :
			active(active) {
	}

	void tic();
	void toc();
	void toc(const string& str);

	void getNsecsRaw(long& sec, long& nsec) const;
	std::string getNsecsFormatted() const;

	long getUsecsRaw() const;
	std::string getUsecsFormatted() const;

	long getMillisRaw() const;
	std::string getMillisFormatted() const;

	long getSecsRaw() const;
	std::string getSecsFormatted() const;

	void print(const string& str) {
		if (active) {

			string s;
			std::stringstream ss;
			ss << setw(9) << setfill('0') << nsec;
			s = ss.str();
			for (int i = s.size() - 3; i > 0; i -= 3) {
				s.insert(s.begin() + i, ',');
			}

			std::cout << str << "Secs: " << sec << " Nsecs: " << setw(11) << s << std::endl;
		}
	}
};

std::ostream& operator<<(std::ostream& out, const TicToc& f) {
	string s;
	std::stringstream ss;
	ss << setw(9) << setfill('0') << f.nsec;
	s = ss.str();
	for (int i = s.size() - 3; i > 0; i -= 3) {
		s.insert(s.begin() + i, ',');
	}

	return out << "Secs: " << f.sec << " Nsecs: " << setw(11) << s;
}

inline void TicToc::tic() {
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
}

inline void TicToc::toc() {
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	diff();
}

inline void TicToc::toc(const string& str) {
	toc();
	print(str);
}

void TicToc::getNsecsRaw(long& sec, long& nsec) const {
	sec = this->sec;
	nsec = this->nsec;
}

std::string TicToc::getNsecsFormatted() const {
	return "";
}

long TicToc::getUsecsRaw() const {
	return sec * 1000000 + nsec / 1000;
}

std::string TicToc::getUsecsFormatted() const {
	return "";
}

long TicToc::getMillisRaw() const {
	return sec * 1000 + nsec / 1000000;
}
std::string TicToc::getMillisFormatted() const {
	return "";
}

long TicToc::getSecsRaw() const {
	return this->sec;
}

std::string TicToc::getSecsFormatted() const {
	string s;
	std::stringstream ss;
	ss << setw(6) << setfill('0') << this->sec;
	s = ss.str();
	for (int i = s.size() - 3; i > 0; i -= 3) {
		s.insert(s.begin() + i, ',');
	}

	return s;
}

void TicToc::diff() {
	if ((end.tv_nsec - start.tv_nsec) < 0) {
		sec = end.tv_sec - start.tv_sec - 1;
		nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	} else {
		sec = end.tv_sec - start.tv_sec;
		nsec = end.tv_nsec - start.tv_nsec;
	}
}
