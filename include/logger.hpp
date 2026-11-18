/*
 * Copyright (C) 2025 André Löfgren
 *
 * This file is part of Biceps.
 *
 * Biceps is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Biceps is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Biceps. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once
#include <iostream>
#include <string>

class Logger {
private:
    int level;
    std::ostream *output_stream;

public:
    Logger();

    Logger(int level, std::ostream &stream);

    int log_level();
    void set_log_level(int value);
    void set_stream(std::ostream &stream);
    void log_msg(std::string msg, int msg_level);
};
