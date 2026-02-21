#include <stdexcept>
#include <format>
#include <sstream>

#include "src/io/csv.h"

namespace io {
    CSVReader::CSVReader(
        const std::string& path
    ):
        m_path(path),
        m_file(path) {
        
        std::string line;
        // store offsets of each row
        while (m_file.good()) {
            m_row_offsets.push_back(m_file.tellg());
            std::getline(m_file, line);
        }
    
    }

    CSVReader::~CSVReader() {
        m_file.close();
    }

    size_t CSVReader::size() const {
        return m_row_offsets.size() - 2;
    }
    
    std::string CSVReader::read_line(
        size_t index
    ) {
        size_t index_after_header = index+1;
        
        if (index_after_header >= size()+1) {
            throw std::out_of_range(std::format("Cannot read line {}. '{}' is of length {}.", index, m_path, size()));
        }
        
        std::string line;
        m_file.clear(); // clear EOF flag
        m_file.seekg(m_row_offsets[index_after_header]);
        std::getline(m_file, line);
        return line;
    }
}
