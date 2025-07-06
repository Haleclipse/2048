/**
 * Framework for 2048 & 2048-Like Games (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Hung Guei
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include <limits>
#include <vector>
#include <cmath>
#include <deque>
#include "board.h"
#include "action.h"
#include "weight.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
};

/**
 * default random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }), popup(0, 9) {}

	virtual action take_action(const board& after) {
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;
			board::cell tile = popup(engine) ? 1 : 2;
			return action::place(pos, tile);
		}
		return action();
	}

private:
	std::array<int, 16> space;
	std::uniform_int_distribution<int> popup;
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

/**
 * 特殊规则的智能玩家：两个8192即胜利
 * 继承weight_agent，实现基于权重的智能决策和避免胜利策略
 */
class strategic_slider : public weight_agent {
private:
	int game_count = 0;
	int move_count = 0;
	std::string last_game_record;
	float danger_penalty_factor = 0.7f;  // 危险惩罚系数
	float survival_bonus = 1000.0f;      // 存活奖励
	std::array<int, 4> opcode;           // 动作顺序
	
	// TD学习相关参数
	float lambda = 0.9f;                 // 折扣因子
	float eligibility_decay = 0.8f;      // 资格迹衰减
	std::vector<std::vector<float>> eligibility_traces; // 资格迹存储
	bool enable_learning = true;         // 是否启用学习
	
	// 游戏轨迹存储
	struct GameStep {
		board state;
		action action_taken;
		board::reward reward;
		board next_state;
		float evaluation;
	};
	std::vector<GameStep> current_episode; // 当前游戏的轨迹
	
public:
	strategic_slider(const std::string& args = "") : weight_agent("name=strategic role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {
		// 解析特殊参数
		if (meta.find("penalty") != meta.end())
			danger_penalty_factor = float(meta["penalty"]);
		if (meta.find("bonus") != meta.end())
			survival_bonus = float(meta["bonus"]);
		if (meta.find("lambda") != meta.end())
			lambda = float(meta["lambda"]);
		if (meta.find("decay") != meta.end())
			eligibility_decay = float(meta["decay"]);
		if (meta.find("learning") != meta.end()) {
			std::string learning_str = meta["learning"];
			enable_learning = (learning_str == "1" || learning_str == "true");
		}
		
		// 初始化资格迹
		initialize_eligibility_traces();
	}

	virtual void open_episode(const std::string& flag = "") override {
		weight_agent::open_episode(flag);
		game_count++;
		move_count = 0;
		last_game_record = "游戏 " + std::to_string(game_count) + " 开始\n";
		
		// 重置游戏轨迹和资格迹
		current_episode.clear();
		reset_eligibility_traces();
	}

	virtual action take_action(const board& before) override {
		move_count++;
		
		// 记录当前盘面状态
		int count_8192 = before.count_tile_value(13);
		int count_4096 = before.count_tile_value(12);
		int max_tile = before.max_tile_value();
		float danger = before.calculate_danger_level();
		
		// 详细状态记录
		last_game_record += "第" + std::to_string(move_count) + "步: ";
		last_game_record += "8192数量=" + std::to_string(count_8192) + " ";
		last_game_record += "4096数量=" + std::to_string(count_4096) + " ";
		last_game_record += "最大瓦片=2^" + std::to_string(max_tile) + " ";
		last_game_record += "危险度=" + std::to_string(danger) + "\n";
		
		// 记录重要时刻的棋盘状态
		bool should_record_board = false;
		
		// 在以下情况记录棋盘状态：
		if (danger > 0.3f) {
			last_game_record += "【危险状态】";
			should_record_board = true;
		} else if (max_tile >= 8) { // 达到256或更高
			last_game_record += "【重要状态】";
			should_record_board = true;
		} else if (move_count % 50 == 0) { // 每50步记录一次
			last_game_record += "【周期记录】";
			should_record_board = true;
		}
		
		if (should_record_board) {
			last_game_record += "当前盘面:\n";
			std::ostringstream oss;
			oss << before;
			last_game_record += oss.str() + "\n";
		}
		
		// 智能决策：评估所有可能的动作
		action selected_action = select_best_action(before);
		
		// 如果不是第一步，对上一步进行TD学习更新
		if (enable_learning && !current_episode.empty()) {
			perform_td_update(before);
		}
		
		return selected_action;
	}

	virtual bool check_for_win(const board& b) override {
		bool has_win = b.has_two_8192();
		if (has_win) {
			last_game_record += "【胜利条件达成】发现两个8192瓦片！\n";
			save_game_record(true);
		}
		return has_win;
	}
	
	virtual void close_episode(const std::string& flag = "") override {
		// 执行最终的TD学习更新
		if (enable_learning && !current_episode.empty()) {
			perform_final_td_update(flag);
		}
		
		// 显示游戏摘要信息
		if (game_count % 50 == 0) {
			show_learning_summary(flag);
		}
		
		last_game_record += "游戏结束，总共" + std::to_string(move_count) + "步\n";
		last_game_record += "结果: " + flag + "\n";
		
		// 记录最终棋盘状态（通过episode获取）
		// 注意：这里我们无法直接获取最终棋盘，所以先注释掉
		// 可以在后续版本中通过其他方式获取
		last_game_record += "\n";
		
		// 保存所有游戏记录（每10局保存一次以避免频繁IO）
		if (game_count % 10 == 0) {
			save_game_record(false);
		}
		
		weight_agent::close_episode(flag);
	}

private:
	/**
	 * 选择最佳动作：结合权重网络评估和避免胜利策略
	 */
	action select_best_action(const board& before) {
		action best_action;
		float best_value = -std::numeric_limits<float>::max();
		
		// 评估所有可能的动作
		for (int op : opcode) {
			board after = before;
			board::reward reward = after.slide(op);
			if (reward == -1) continue; // 无效动作
			
			// 计算这个动作的评估值
			float value = evaluate_action(before, after, reward);
			
			if (value > best_value) {
				best_value = value;
				best_action = action::slide(op);
			}
		}
		
		// 如果没有找到有效动作，使用随机策略作为后备
		if (best_value == -std::numeric_limits<float>::max()) {
			for (int op : opcode) {
				board after = before;
				if (after.slide(op) != -1) {
					return action::slide(op);
				}
			}
		}
		
		// 记录选择的动作到游戏轨迹中
		if (enable_learning && best_action.type() != 0) {
			// 执行最佳动作获取下一状态
			board next_state = before;
			board::reward actual_reward = best_action.apply(next_state);
			
			// 存储游戏步骤
			GameStep step;
			step.state = before;
			step.action_taken = best_action;
			step.reward = actual_reward;
			step.next_state = next_state;
			step.evaluation = evaluate_board(before);
			
			current_episode.push_back(step);
		}
		
		return best_action;
	}
	
	/**
	 * 评估一个动作的价值：基础价值 + 策略调整
	 */
	float evaluate_action(const board& before, const board& after, board::reward reward) {
		// 基础评估：合并奖励 + 权重网络评估（如果有的话）
		float base_value = static_cast<float>(reward);
		
		// 如果网络已初始化，使用网络评估
		if (!net.empty()) {
			base_value += evaluate_board(after);
		}
		
		// 策略调整：考虑危险度
		float danger_after = after.calculate_danger_level();
		float strategy_penalty = danger_after * danger_penalty_factor * 10000.0f;
		
		// 存活奖励：更多空格 = 更好
		int empty_cells = 0;
		for (int i = 0; i < 16; i++) {
			if (after(i) == 0) empty_cells++;
		}
		float survival_reward = empty_cells * survival_bonus;
		
		// 最终评估值
		float final_value = base_value - strategy_penalty + survival_reward;
		
		return final_value;
	}
	
	/**
	 * 使用完整的N-tuple网络评估棋盘状态
	 * 实现多个重叠的瓦片模式进行特征提取
	 */
	float evaluate_board(const board& b) {
		if (net.empty()) return 0.0f;
		
		float value = 0.0f;
		
		// 定义常用的N-tuple模式（4-tuple设计）
		// 使用4x4-tuple模式，每个模式覆盖4个相邻位置
		static const std::vector<std::vector<int>> patterns = {
			{0, 1, 4, 5},   // 左上角2x2
			{2, 3, 6, 7},   // 右上角2x2  
			{8, 9, 12, 13}, // 左下角2x2
			{10, 11, 14, 15} // 右下角2x2
		};
		
		// 对每个模式和其同构变换进行评估
		for (size_t i = 0; i < std::min(patterns.size(), net.size()); i++) {
			if (net[i].size() == 0) continue;
			
			// 评估原始模式
			value += evaluate_pattern(b, patterns[i], net[i]);
			
			// 评估同构变换（镜像、旋转等）
			value += evaluate_isomorphic_patterns(b, patterns[i], net[i]);
		}
		
		return value;
	}
	
	/**
	 * 评估单个N-tuple模式
	 */
	float evaluate_pattern(const board& b, const std::vector<int>& pattern, const weight& weights) {
		// 计算模式索引
		size_t index = 0;
		size_t multiplier = 1;
		
		for (int pos : pattern) {
			if (pos >= 0 && pos < 16) {
				// 获取瓦片值（对数形式），限制最大值为15（避免索引越界）
				int tile_value = std::min(static_cast<int>(b(pos)), 15);
				index += tile_value * multiplier;
				multiplier *= 16; // 每个位置最多16种可能值（0-15）
			}
			
			// 防止索引过大
			if (index >= weights.size()) {
				break;
			}
		}
		
		// 确保索引在有效范围内
		if (index < weights.size()) {
			return weights[index];
		}
		
		return 0.0f;
	}
	
	/**
	 * 评估模式的同构变换（镜像和旋转）
	 */
	float evaluate_isomorphic_patterns(const board& b, const std::vector<int>& pattern, const weight& weights) {
		float total_value = 0.0f;
		
		// 水平镜像
		board mirrored_h = b;
		mirror_horizontal(mirrored_h);
		total_value += evaluate_pattern(mirrored_h, pattern, weights);
		
		// 垂直镜像
		board mirrored_v = b;
		mirror_vertical(mirrored_v);
		total_value += evaluate_pattern(mirrored_v, pattern, weights);
		
		// 转置
		board transposed = b;
		transpose_board(transposed);
		total_value += evaluate_pattern(transposed, pattern, weights);
		
		// 转置后镜像
		mirror_horizontal(transposed);
		total_value += evaluate_pattern(transposed, pattern, weights);
		
		mirror_vertical(transposed);
		total_value += evaluate_pattern(transposed, pattern, weights);
		
		// 再次转置
		transpose_board(transposed);
		total_value += evaluate_pattern(transposed, pattern, weights);
		
		mirror_horizontal(transposed);
		total_value += evaluate_pattern(transposed, pattern, weights);
		
		return total_value;
	}
	
	/**
	 * 棋盘变换函数
	 */
	void mirror_horizontal(board& b) {
		for (int r = 0; r < 4; r++) {
			std::swap(b(r*4 + 0), b(r*4 + 3));
			std::swap(b(r*4 + 1), b(r*4 + 2));
		}
	}
	
	void mirror_vertical(board& b) {
		for (int c = 0; c < 4; c++) {
			std::swap(b(0*4 + c), b(3*4 + c));
			std::swap(b(1*4 + c), b(2*4 + c));
		}
	}
	
	void transpose_board(board& b) {
		for (int r = 0; r < 4; r++) {
			for (int c = r + 1; c < 4; c++) {
				std::swap(b(r*4 + c), b(c*4 + r));
			}
		}
	}

	void save_game_record(bool is_win) {
		std::string filename = is_win ? "win_games.log" : "normal_games.log";
		std::ofstream file(filename, std::ios::app);
		if (file.is_open()) {
			file << "=== " << (is_win ? "胜利游戏" : "普通游戏") << " ===\n";
			file << last_game_record;
			file << "================================\n\n";
			file.close();
		}
	}
	
	/**
	 * TD学习相关方法
	 */
	
	// 初始化资格迹
	void initialize_eligibility_traces() {
		if (net.empty()) return;
		
		eligibility_traces.resize(net.size());
		for (size_t i = 0; i < net.size(); i++) {
			eligibility_traces[i].resize(net[i].size(), 0.0f);
		}
	}
	
	// 重置资格迹
	void reset_eligibility_traces() {
		for (auto& traces : eligibility_traces) {
			std::fill(traces.begin(), traces.end(), 0.0f);
		}
	}
	
	// 衰减资格迹
	void decay_eligibility_traces() {
		for (auto& traces : eligibility_traces) {
			for (auto& trace : traces) {
				trace *= eligibility_decay;
			}
		}
	}
	
	// 执行TD学习更新
	void perform_td_update(const board& current_state) {
		if (current_episode.size() < 2) return;
		
		// 获取上一步的信息
		GameStep& prev_step = current_episode[current_episode.size() - 2];
		
		// 计算TD目标和误差
		float current_value = evaluate_board(current_state);
		float td_target = prev_step.reward + lambda * current_value;
		float td_error = td_target - prev_step.evaluation;
		
		// 更新权重
		update_weights_with_td_error(prev_step.state, td_error);
		
		// 衰减资格迹
		decay_eligibility_traces();
		
		// 记录TD学习统计
		static float total_td_error = 0.0f;
		static int td_update_count = 0;
		
		total_td_error += std::abs(td_error);
		td_update_count++;
		
		// 每100次更新输出一次学习统计
		if (td_update_count % 100 == 0) {
			float avg_td_error = total_td_error / 100;
			std::cout << " [TD: 误差=" << std::fixed << std::setprecision(2) << avg_td_error 
			          << " 学习率=" << alpha << "]" << std::flush;
			total_td_error = 0.0f;
			td_update_count = 0;
		}
		
		// 如果启用详细日志，记录TD更新信息
		if (game_count % 100 == 0 && move_count % 50 == 0) {
			last_game_record += "【TD更新】TD误差=" + std::to_string(td_error) + 
			                   ", 当前值=" + std::to_string(current_value) + "\n";
		}
	}
	
	// 游戏结束时的最终TD更新
	void perform_final_td_update(const std::string& flag) {
		if (current_episode.empty()) return;
		
		// 获取最后一步
		GameStep& last_step = current_episode.back();
		
		// 游戏结束时的最终奖励
		float final_reward = calculate_final_reward(flag);
		
		// 计算TD误差（终止状态的值为0）
		float td_error = final_reward - last_step.evaluation;
		
		// 对整个游戏轨迹进行反向传播更新
		for (int i = current_episode.size() - 1; i >= 0; i--) {
			GameStep& step = current_episode[i];
			
			// 计算折扣后的TD误差
			float discounted_error = td_error * std::pow(lambda, current_episode.size() - 1 - i);
			
			// 更新权重
			update_weights_with_td_error(step.state, discounted_error);
			
			// 根据步骤奖励调整TD误差
			if (i > 0) {
				td_error = step.reward + lambda * td_error;
			}
		}
		
		// 记录最终更新信息
		last_game_record += "【最终TD更新】游戏长度=" + std::to_string(current_episode.size()) + 
		                   "步, 最终奖励=" + std::to_string(final_reward) + "\n";
	}
	
	// 使用TD误差更新权重
	void update_weights_with_td_error(const board& state, float td_error) {
		if (net.empty() || eligibility_traces.empty()) return;
		
		// 定义N-tuple模式
		static const std::vector<std::vector<int>> patterns = {
			{0, 1, 4, 5},   // 左上角2x2
			{2, 3, 6, 7},   // 右上角2x2  
			{8, 9, 12, 13}, // 左下角2x2
			{10, 11, 14, 15} // 右下角2x2
		};
		
		// 对每个模式进行权重更新
		for (size_t i = 0; i < std::min(patterns.size(), net.size()); i++) {
			if (net[i].size() == 0) continue;
			
			// 更新原始模式
			update_pattern_weights(state, patterns[i], i, td_error);
			
			// 更新同构变换
			update_isomorphic_pattern_weights(state, patterns[i], i, td_error);
		}
	}
	
	// 更新单个模式的权重
	void update_pattern_weights(const board& state, const std::vector<int>& pattern, 
	                           size_t net_index, float td_error) {
		// 计算模式索引
		size_t index = 0;
		size_t multiplier = 1;
		
		for (int pos : pattern) {
			if (pos >= 0 && pos < 16) {
				int tile_value = std::min(static_cast<int>(state(pos)), 15);
				index += tile_value * multiplier;
				multiplier *= 16;
			}
			
			if (index >= net[net_index].size()) break;
		}
		
		// 更新权重和资格迹
		if (index < net[net_index].size() && index < eligibility_traces[net_index].size()) {
			// 设置当前状态的资格迹为1
			eligibility_traces[net_index][index] = 1.0f;
			
			// TD(λ)权重更新
			net[net_index][index] += alpha * td_error * eligibility_traces[net_index][index];
		}
	}
	
	// 更新同构变换模式的权重
	void update_isomorphic_pattern_weights(const board& state, const std::vector<int>& pattern,
	                                     size_t net_index, float td_error) {
		// 为了简化，这里只实现水平镜像的权重更新
		// 完整版本应该包含所有8种变换
		
		board mirrored = state;
		mirror_horizontal(mirrored);
		update_pattern_weights(mirrored, pattern, net_index, td_error * 0.125f); // 8种变换均分
		
		board transposed = state;
		transpose_board(transposed);
		update_pattern_weights(transposed, pattern, net_index, td_error * 0.125f);
	}
	
	// 计算游戏结束时的最终奖励
	float calculate_final_reward(const std::string& flag) {
		float final_reward = 0.0f;
		
		if (flag == "win") {
			// 胜利：这在我们的特殊规则下是不好的结果
			final_reward = -50000.0f;
		} else if (flag == "lose") {
			// 失败：但如果避免了胜利且得分较高，给予奖励
			if (!current_episode.empty()) {
				board final_state = current_episode.back().next_state;
				int count_8192 = final_state.count_tile_value(13);
				
				if (count_8192 == 1) {
					// 成功维持一个8192而没有胜利
					final_reward = 10000.0f;
				} else if (count_8192 == 0 && final_state.max_tile_value() >= 12) {
					// 至少达到4096
					final_reward = 5000.0f;
				} else {
					final_reward = 1000.0f; // 基础完成奖励
				}
			}
		}
		
		return final_reward;
	}
	
	// 提取所有特征用于权重更新
	std::vector<std::vector<size_t>> extract_all_features(const board& state) {
		std::vector<std::vector<size_t>> all_features;
		
		static const std::vector<std::vector<int>> patterns = {
			{0, 1, 4, 5}, {2, 3, 6, 7}, {8, 9, 12, 13}, {10, 11, 14, 15}
		};
		
		for (const auto& pattern : patterns) {
			std::vector<size_t> features;
			
			// 计算原始模式索引
			size_t index = 0;
			size_t multiplier = 1;
			
			for (int pos : pattern) {
				if (pos >= 0 && pos < 16) {
					int tile_value = std::min(static_cast<int>(state(pos)), 15);
					index += tile_value * multiplier;
					multiplier *= 16;
				}
			}
			
			features.push_back(index);
			all_features.push_back(features);
		}
		
		return all_features;
	}
	
public:
	// 公共接口用于调整学习参数
	void set_learning_rate(float new_alpha) { alpha = new_alpha; }
	void set_lambda(float new_lambda) { lambda = new_lambda; }
	void set_danger_penalty_factor(float new_penalty) { danger_penalty_factor = new_penalty; }
	void set_survival_bonus(float new_bonus) { survival_bonus = new_bonus; }
	void set_learning_enabled(bool enabled) { enable_learning = enabled; }
	
	// 获取学习统计信息
	size_t get_episode_length() const { return current_episode.size(); }
	float get_current_learning_rate() const { return alpha; }
	bool is_learning_enabled() const { return enable_learning; }
	
private:
	// 显示学习摘要
	void show_learning_summary(const std::string& flag) {
		if (!enable_learning) return;
		
		static int win_count = 0;
		static int lose_count = 0;
		static int total_steps = 0;
		static float total_danger = 0.0f;
		
		// 统计游戏结果
		if (flag == "win") win_count++;
		else lose_count++;
		
		total_steps += move_count;
		
		// 计算平均危险度
		if (!current_episode.empty()) {
			float avg_danger = 0.0f;
			for (const auto& step : current_episode) {
				avg_danger += step.state.calculate_danger_level();
			}
			total_danger += avg_danger / current_episode.size();
		}
		
		// 每50局输出摘要
		if (game_count % 50 == 0) {
			float avg_steps = float(total_steps) / 50;
			float avg_danger = total_danger / 50;
			float win_rate = float(win_count) / (win_count + lose_count) * 100;
			
			std::cout << "\n[学习摘要] 游戏" << (game_count-49) << "-" << game_count 
			          << ": 平均步数=" << int(avg_steps) 
			          << " 胜利避免率=" << std::fixed << std::setprecision(1) << (100-win_rate) << "%" 
			          << " 平均危险度=" << std::setprecision(3) << avg_danger 
			          << " 学习率=" << alpha << std::endl;
			
			// 重置统计
			win_count = lose_count = total_steps = 0;
			total_danger = 0.0f;
		}
	}
};
