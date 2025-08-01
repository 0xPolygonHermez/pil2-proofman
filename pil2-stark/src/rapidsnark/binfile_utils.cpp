#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <system_error>
#include <string>
#include <memory.h>
#include <stdexcept>

#include "binfile_utils.hpp"
#include "thread_utils.hpp"
#include <omp.h>

namespace BinFileUtils
{
    BinFile::BinFile(void *data, uint64_t _size, std::string _type, uint32_t maxVersion)
        : addr(nullptr), size(0), pos(0), version(0), readingSection(nullptr)
    {
        size = _size;
        addr = malloc(size);
        if (addr == nullptr) {
            throw std::bad_alloc();
        }
        
        int nThreads = omp_get_max_threads() / 2;
        ThreadUtils::parcpy(addr, data, size, nThreads);
        
        type.assign((const char *)addr, 4);
        pos = 4;

        if (type != _type)
        {
            throw std::invalid_argument("Invalid file type. It should be " + _type + " and it us " + type);
        }

        version = readU32LE();
        if (version > maxVersion)
        {
            throw std::invalid_argument("Invalid version. It should be <=" + std::to_string(maxVersion) + " and it us " + std::to_string(version));
        }

        u_int32_t nSections = readU32LE();

        for (u_int32_t i = 0; i < nSections; i++)
        {
            u_int32_t sType = readU32LE();
            u_int64_t sSize = readU64LE();

            if (sections.find(sType) == sections.end())
            {
                sections.insert(std::make_pair(sType, std::vector<Section>()));
            }

            sections[sType].push_back(Section((void *)((u_int64_t)addr + pos), sSize));

            pos += sSize;
        }

        pos = 0;
        readingSection = nullptr;
    }

    BinFile::BinFile(std::string fileName, std::string _type, uint32_t maxVersion)
        : addr(nullptr), size(0), pos(0), version(0), readingSection(nullptr)
    {
        
        int fd;
        struct stat sb;

        fd = open(fileName.c_str(), O_RDONLY);
        if (fd == -1)
            throw std::system_error(errno, std::generic_category(), "open");

        if (fstat(fd, &sb) == -1) /* To obtain file size */
            throw std::system_error(errno, std::generic_category(), "fstat");

        size = sb.st_size + sizeof(u_int32_t) + sizeof(u_int64_t);
        void *addrmm = mmap(NULL, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (addrmm == MAP_FAILED) {
            close(fd);
            throw std::system_error(errno, std::generic_category(), "mmap");
        }
        
        addr = malloc(size);
        if (addr == nullptr) {
            munmap(addrmm, size);
            close(fd);
            throw std::bad_alloc();
        }
        
        // int nThreads = omp_get_max_threads() / 2;
        // ThreadUtils::parcpy(addr, addrmm, size, nThreads);
        memcpy(addr, addrmm, sb.st_size);

        munmap(addrmm, size);
        close(fd);

        type.assign((const char *)addr, 4);
        pos = 4;

        if (type != _type)
        {
            throw std::invalid_argument("Invalid file type. It should be " + _type + " and it us " + type);
        }

        version = readU32LE();
        if (version > maxVersion)
        {
            throw std::invalid_argument("Invalid version. It should be <=" + std::to_string(maxVersion) + " and it us " + std::to_string(version));
        }

        u_int32_t nSections = readU32LE();

        for (u_int32_t i = 0; i < nSections; i++)
        {
            u_int32_t sType = readU32LE();
            u_int64_t sSize = readU64LE();

            if (sections.find(sType) == sections.end())
            {
                sections.insert(std::make_pair(sType, std::vector<Section>()));
            }

            sections[sType].push_back(Section((void *)((u_int64_t)addr + pos), sSize));

            pos += sSize;
        }

        pos = 0;
        readingSection = nullptr;
    }

    BinFile::~BinFile()
    {
        free(addr);
    }

    void BinFile::startReadSection(u_int32_t sectionId, u_int32_t sectionPos)
    {
        if (sections.find(sectionId) == sections.end())
        {
            throw std::range_error("Section does not exist: " + std::to_string(sectionId));
        }

        if (sectionPos >= sections[sectionId].size())
        {
            throw std::range_error("Section pos too big. There are " + std::to_string(sections[sectionId].size()) + " and it's trying to access section: " + std::to_string(sectionPos));
        }

        if (readingSection != nullptr)
        {
            throw std::range_error("Already reading a section");
        }

        pos = (u_int64_t)(sections[sectionId][sectionPos].start) - (u_int64_t)addr;

        readingSection = &sections[sectionId][sectionPos];
    }

    void BinFile::endReadSection(bool check)
    {
        if (check)
        {
            if ((u_int64_t)addr + pos - (u_int64_t)(readingSection->start) != readingSection->size)
            {
                throw std::range_error("Invalid section size");
            }
        }
        readingSection = nullptr;
    }

    void *BinFile::getSectionData(u_int32_t sectionId, u_int32_t sectionPos)
    {

        if (sections.find(sectionId) == sections.end())
        {
            throw std::range_error("Section does not exist: " + std::to_string(sectionId));
        }

        if (sectionPos >= sections[sectionId].size())
        {
            throw std::range_error("Section pos too big. There are " + std::to_string(sections[sectionId].size()) + " and it's trying to access section: " + std::to_string(sectionPos));
        }

        return sections[sectionId][sectionPos].start;
    }

    u_int64_t BinFile::getSectionSize(u_int32_t sectionId, u_int32_t sectionPos)
    {

        if (sections.find(sectionId) == sections.end())
        {
            throw std::range_error("Section does not exist: " + std::to_string(sectionId));
        }

        if (sectionPos >= sections[sectionId].size())
        {
            throw std::range_error("Section pos too big. There are " + std::to_string(sections[sectionId].size()) + " and it's trying to access section: " + std::to_string(sectionPos));
        }

        return sections[sectionId][sectionPos].size;
    }

    u_int8_t BinFile::readU8LE()
    {
        if (pos + sizeof(u_int8_t) > size) {
            throw std::out_of_range("Attempting to read beyond buffer bounds");
        }
        u_int8_t res = *((u_int8_t *)((u_int64_t)addr + pos));
        pos += 1;
        return res;
    }


    u_int16_t BinFile::readU16LE()
    {
        if (pos + sizeof(u_int16_t) > size) {
            throw std::out_of_range("Attempting to read beyond buffer bounds");
        }
        u_int16_t res = *((u_int16_t *)((u_int64_t)addr + pos));
        pos += 2;
        return res;
    }


    u_int32_t BinFile::readU32LE()
    {
        if (pos + sizeof(u_int32_t) > size) {
            throw std::out_of_range("Attempting to read beyond buffer bounds");
        }
        u_int32_t res = *((u_int32_t *)((u_int64_t)addr + pos));
        pos += 4;
        return res;
    }

    u_int64_t BinFile::readU64LE()
    {
        if (pos + sizeof(u_int64_t) > size) {
            throw std::out_of_range("Attempting to read beyond buffer bounds u64");
        }
        u_int64_t res = *((u_int64_t *)((u_int64_t)addr + pos));
        pos += 8;
        return res;
    }

    bool BinFile::sectionExists(u_int32_t sectionId) {
        return sections.find(sectionId) != sections.end();
    }

    void *BinFile::read(u_int64_t len)
    {
        if (pos + len > size) {
            throw std::out_of_range("Attempting to read beyond buffer bounds in read()");
        }
        void *res = (void *)((u_int64_t)addr + pos);
        pos += len;
        return res;
    }

    std::string BinFile::readString()
    {
        uint8_t *startOfString = (uint8_t *)((u_int64_t)addr + pos);
        uint8_t *endOfString = startOfString;
        uint8_t *endOfSection = (uint8_t *)((uint64_t)readingSection->start + readingSection->size);

        uint8_t *i;
        for (i = endOfString; i != endOfSection; i++)
        {
            if (*i == 0)
            {
                endOfString = i;
                break;
            }
        }

        if (i == endOfSection)
        {
            endOfString = i - 1;
        }

        uint32_t len = endOfString - startOfString;
        std::string str = std::string((const char *)startOfString, len);
        pos += len + 1;

        return str;
    }

    std::unique_ptr<BinFile> openExisting(std::string filename, std::string type, uint32_t maxVersion)
    {
        return std::unique_ptr<BinFile>(new BinFile(filename, type, maxVersion));
    }

} // Namespace