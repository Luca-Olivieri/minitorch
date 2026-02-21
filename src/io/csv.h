#ifndef CSV_H
#define CSV_H

#include <fstream>
#include <vector>
#include <string>

namespace io {

    class CSVReader {
    public:
        CSVReader(
                const std::string& path
        );

        ~CSVReader();

        size_t size() const;
    
        std::string read_line(
                size_t index
        );
    
    private:
        const std::string m_path;
        std::vector<std::streampos> m_row_offsets;
        std::ifstream m_file;
    };
}

#endif
