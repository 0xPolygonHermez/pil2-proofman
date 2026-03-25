#ifndef BINFILE_WRITER_H
#define BINFILE_WRITER_H
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <fstream>
#include <cstdint>


namespace BinFileUtils
{
    class BinFileWriter
    {
        std::ofstream file;

        class Section
        {
            off_t start;
            u_int64_t size;

        public:
            friend BinFileWriter;
            Section(off_t _start, u_int64_t _size) : start(_start), size(_size){};
        };

        std::map<int, Section*> sections;

        Section *writingSection;

        off_t getCurrentPosition();

        // Initial values when creating the file
        std::string filename;
        std::string type;
        uint32_t version;
        uint32_t nSections;

    public:
        BinFileWriter(std::string filename, std::string type, uint32_t version, uint32_t nSections);

        ~BinFileWriter();

        void close();
        
        void startWriteSection(u_int32_t sectionId);
        void endWriteSection();

        void writeU8LE(u_int8_t value);
        void writeU16LE(u_int16_t value);
        void writeU32LE(u_int32_t value);
        void writeU64LE(u_int64_t value);
        void write(void *buffer, u_int64_t len);
        void writeString(const std::string& str);
    };
}

#endif // BINFILE_WRITER_H