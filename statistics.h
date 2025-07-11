/**
 * Framework for 2048 & 2048-Like Games (C++ 11)
 * statistics.h: Utility for making statistical reports
 *
 * Author: Hung Guei
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <deque>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "board.h"
#include "action.h"
#include "episode.h"

class statistics {
public:
	/**
	 * the total episodes to run
	 * the block size of statistics
	 * the limit of saving records
	 *
	 * note that total >= limit >= block
	 */
	statistics(size_t total, size_t block = 0, size_t limit = 0)
		: total(total),
		  block(block ? block : total),
		  limit(limit ? limit : total),
		  count(0) {}

public:
	/**
	 * show the statistics of last 'block' games
	 *
	 * the format is
	 * 1000   avg = 273901, max = 382324, ops = 241563 (170543|896715)
	 *        512     100%   (0.3%)
	 *        1024    99.7%  (0.2%)
	 *        2048    99.5%  (1.1%)
	 *        4096    98.4%  (4.7%)
	 *        8192    93.7%  (22.4%)
	 *        16384   71.3%  (71.3%)
	 *
	 * where
	 *  '1000': current index
	 *  'avg = 273901, max = 382324': the average score is 273901, the maximum score is 382324
	 *  'ops = 241563 (170543|896715)': the average speed is 241563
	 *                                  the average speed of the slider is 170543
	 *                                  the average speed of the placer is 896715
	 *  '93.7%': 93.7% of the games reached 8192-tiles, i.e., win rate of 8192-tile
	 *  '22.4%': 22.4% of the games terminated with 8192-tiles as the largest tile
	 */
	void show(bool tstat = true, size_t blk = 0) const {
		size_t num = std::min(data.size(), blk ?: block);
		size_t stat[64] = { 0 };
		size_t sop = 0, pop = 0, eop = 0;
		time_t sdu = 0, pdu = 0, edu = 0;
		board::score sum = 0, max = 0;
		auto it = data.end();
		for (size_t i = 0; i < num; i++) {
			auto& ep = *(--it);
			sum += ep.score();
			max = std::max(ep.score(), max);
			stat[*std::max_element(ep.state().begin(), ep.state().end())]++;
			sop += ep.step();
			pop += ep.step(action::slide::type);
			eop += ep.step(action::place::type);
			sdu += ep.time();
			pdu += ep.time(action::slide::type);
			edu += ep.time(action::place::type);
		}

		std::ios ff(nullptr);
		ff.copyfmt(std::cout);
		std::cout << std::fixed << std::setprecision(0);
		std::cout << count << "\t";
		std::cout << "平均分 = " << (sum / num) << ", ";
		std::cout << "最高分 = " << (max) << ", ";
		std::cout << "ops = " << (sop * 1000.0 / sdu);
		std::cout <<     " (" << (pop * 1000.0 / pdu);
		std::cout <<      "|" << (eop * 1000.0 / edu) << ")";
		std::cout << std::endl;
		std::cout.copyfmt(ff);

		if (!tstat) return;
		for (size_t t = 0, c = 0; c < num; c += stat[t++]) {
			if (stat[t] == 0) continue;
			size_t accu = std::accumulate(std::begin(stat) + t, std::end(stat), size_t(0));
			std::cout << "\t" << ((1 << t) & -2u); // type
			std::cout << "\t" << (accu * 100.0 / num) << "%"; // win rate
			std::cout << "\t" "(" << (stat[t] * 100.0 / num) << "%" ")"; // percentage of ending
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	void summary() const {
		show(true, data.size());
	}
	
	/**
	 * 显示训练进度
	 */
	void show_progress() const {
		if (data.empty()) return;
		
		// 计算最近100局的统计
		size_t recent_count = std::min(size_t(100), data.size());
		board::score sum = 0, max = 0;
		
		for (auto it = data.end() - recent_count; it != data.end(); ++it) {
			sum += it->score();
			max = std::max(max, it->score());
		}
		
		double avg = recent_count ? double(sum) / recent_count : 0.0;
		double progress = double(count) / total * 100.0;
		
		// 输出简洁的进度信息
		std::cout << "\r进度 " << count << "/" << total 
		          << " (" << std::fixed << std::setprecision(1) << progress << "%) "
		          << "平均=" << int(avg) << " 最高=" << max << std::flush;
		
		// 每1000局换行
		if (count % 1000 == 0) {
			std::cout << std::endl;
		}
	}

	bool is_finished() const {
		return count >= total;
	}

	void open_episode(const std::string& flag = "") {
		if (count++ >= limit) data.pop_front();
		data.emplace_back();
		data.back().open_episode(flag);
	}

	void close_episode(const std::string& flag = "") {
		data.back().close_episode(flag);
		
		// 每100局显示简要进度
		if (count % 100 == 0) {
			show_progress();
		}
		
		// 每block显示详细统计
		if (count % block == 0) show();
	}

	episode& at(size_t i) {
		return data.at(i);
	}
	episode& front() {
		return data.front();
	}
	episode& back() {
		return data.back();
	}
	size_t step() const {
		return count;
	}

	friend std::ostream& operator <<(std::ostream& out, const statistics& stat) {
		for (const episode& rec : stat.data) out << rec << std::endl;
		return out;
	}
	friend std::istream& operator >>(std::istream& in, statistics& stat) {
		for (std::string line; std::getline(in, line) && line.size(); ) {
			stat.data.emplace_back();
			std::stringstream(line) >> stat.data.back();
		}
		stat.total = std::max(stat.total, stat.data.size());
		stat.count = stat.data.size();
		return in;
	}

private:
	size_t total;
	size_t block;
	size_t limit;
	size_t count;
	std::deque<episode> data;
};
