/**
 * Framework for 2048 & 2048-Like Games (C++ 11)
 * board.h: Define the game state and basic operations of the game of 2048
 *
 * Author: Hung Guei
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <array>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdint>

/**
 * array-based board for 2048
 *
 * index (1-d form):
 *  (0)  (1)  (2)  (3)
 *  (4)  (5)  (6)  (7)
 *  (8)  (9) (10) (11)
 * (12) (13) (14) (15)
 *
 */
class board {
public:
	typedef uint32_t cell;
	typedef std::array<cell, 4> row;
	typedef std::array<row, 4> grid;
	typedef uint64_t data;
	typedef uint64_t score;
	typedef int reward;

public:
	board() : tile(), attr(0) {}
	board(const grid& b, data v = 0) : tile(b), attr(v) {}
	board(const board& b) = default;
	board& operator =(const board& b) = default;

	operator grid&() { return tile; }
	operator const grid&() const { return tile; }
	row& operator [](unsigned i) { return tile[i]; }
	const row& operator [](unsigned i) const { return tile[i]; }
	cell& operator ()(unsigned i) { return tile[i / 4][i % 4]; }
	const cell& operator ()(unsigned i) const { return tile[i / 4][i % 4]; }

	cell* begin() { return &(operator()(0)); }
	const cell* begin() const { return &(operator()(0)); }
	cell* end() { return begin() + 16; }
	const cell* end() const { return begin() + 16; }

	data info() const { return attr; }
	data info(data dat) { data old = attr; attr = dat; return old; }

public:
	bool operator ==(const board& b) const { return tile == b.tile; }
	bool operator < (const board& b) const { return tile <  b.tile; }
	bool operator !=(const board& b) const { return !(*this == b); }
	bool operator > (const board& b) const { return b < *this; }
	bool operator <=(const board& b) const { return !(b < *this); }
	bool operator >=(const board& b) const { return !(*this < b); }

public:

	/**
	 * place a tile (index value) to the specific position (1-d index)
	 * return 0 if the action is valid, or -1 if not
	 */
	reward place(unsigned pos, cell tile) {
		if (pos >= 16 || operator()(pos)) return -1;
		if (tile != 1 && tile != 2) return -1;
		operator()(pos) = tile;
		return 0;
	}

	/**
	 * apply an action to the board
	 * return the reward of the action, or -1 if the action is illegal
	 */
	reward slide(unsigned opcode) {
		switch (opcode & 0b11) {
		case 0: return slide_up();
		case 1: return slide_right();
		case 2: return slide_down();
		case 3: return slide_left();
		default: return -1;
		}
	}

	reward slide_left() {
		board prev = *this;
		reward score = 0;
		for (int r = 0; r < 4; r++) {
			auto& row = tile[r];
			int top = 0, hold = 0;
			for (int c = 0; c < 4; c++) {
				int tile = row[c];
				if (tile == 0) continue;
				row[c] = 0;
				if (hold) {
					if (tile == hold) {
						row[top++] = ++tile;
						score += (1 << tile);
						hold = 0;
					} else {
						row[top++] = hold;
						hold = tile;
					}
				} else {
					hold = tile;
				}
			}
			if (hold) tile[r][top] = hold;
		}
		return (*this != prev) ? score : -1;
	}
	reward slide_right() {
		reflect_horizontal();
		reward score = slide_left();
		reflect_horizontal();
		return score;
	}
	reward slide_up() {
		rotate_clockwise();
		reward score = slide_right();
		rotate_counterclockwise();
		return score;
	}
	reward slide_down() {
		rotate_clockwise();
		reward score = slide_left();
		rotate_counterclockwise();
		return score;
	}

	void rotate(int clockwise_count = 1) {
		switch (((clockwise_count % 4) + 4) % 4) {
		default:
		case 0: break;
		case 1: rotate_clockwise(); break;
		case 2: reverse(); break;
		case 3: rotate_counterclockwise(); break;
		}
	}

	void rotate_clockwise() { transpose(); reflect_horizontal(); }
	void rotate_counterclockwise() { transpose(); reflect_vertical(); }
	void reverse() { reflect_horizontal(); reflect_vertical(); }

	void reflect_horizontal() {
		for (int r = 0; r < 4; r++) {
			std::swap(tile[r][0], tile[r][3]);
			std::swap(tile[r][1], tile[r][2]);
		}
	}

	void reflect_vertical() {
		for (int c = 0; c < 4; c++) {
			std::swap(tile[0][c], tile[3][c]);
			std::swap(tile[1][c], tile[2][c]);
		}
	}

	void transpose() {
		for (int r = 0; r < 4; r++) {
			for (int c = r + 1; c < 4; c++) {
				std::swap(tile[r][c], tile[c][r]);
			}
		}
	}

public:
	/**
	 * 检测是否有两个8192瓦片 (胜利条件)
	 * 8192 = 2^13, 所以检测值为13
	 */
	bool has_two_8192() const {
		int count_8192 = 0;
		for (int i = 0; i < 16; i++) {
			if (operator()(i) == 13) { // 2^13 = 8192
				count_8192++;
			}
		}
		return count_8192 >= 2;
	}
	
	/**
	 * 计算特定数值瓦片的数量
	 */
	int count_tile_value(cell value) const {
		int count = 0;
		for (int i = 0; i < 16; i++) {
			if (operator()(i) == value) {
				count++;
			}
		}
		return count;
	}
	
	/**
	 * 获取最大瓦片值
	 */
	int max_tile_value() const {
		int max_val = 0;
		for (int i = 0; i < 16; i++) {
			max_val = std::max(max_val, static_cast<int>(operator()(i)));
		}
		return max_val;
	}
	
	/**
	 * 计算危险程度 (接近胜利条件的程度)
	 * 返回值: 0.0 = 安全, 1.0 = 极度危险
	 */
	float calculate_danger_level() const {
		int count_8192 = count_tile_value(13); // 8192
		int count_4096 = count_tile_value(12); // 4096
		
		if (count_8192 >= 1 && count_4096 >= 2) return 1.0f; // 极高危险
		if (count_8192 >= 1 && count_4096 >= 1) return 0.7f; // 高危险
		if (count_4096 >= 3) return 0.4f; // 中等危险
		return 0.0f; // 安全
	}

public:
	friend std::ostream& operator <<(std::ostream& out, const board& b) {
		out << "+------------------------+" << std::endl;
		for (auto& row : b.tile) {
			out << "|" << std::dec;
			for (auto t : row) out << std::setw(6) << ((1 << t) & -2u);
			out << "|" << std::endl;
		}
		out << "+------------------------+" << std::endl;
		return out;
	}
	friend std::istream& operator >>(std::istream& in, board& b) {
		for (int i = 0; i < 16; i++) {
			while (!std::isdigit(in.peek()) && in.good()) in.ignore(1);
			in >> b(i);
			b(i) = std::log2(b(i));
		}
		return in;
	}

private:
	grid tile;
	data attr;
};
