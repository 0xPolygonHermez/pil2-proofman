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
#include <thread>
#include <vector>
#include <algorithm>

namespace BinFileUtils
{
    BinFile::BinFile(void *data, uint64_t _size, std::string _type, uint32_t maxVersion)
        : addr(nullptr), size(0), pos(0), directRead(false), fileFd(-1), version(0), readingSection(nullptr)
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
        : addr(nullptr), size(0), pos(0), directRead(false), fileFd(-1), version(0), readingSection(nullptr)
    {
        
        int fd;
        struct stat sb;

        fd = open(fileName.c_str(), O_RDONLY);
        if (fd == -1)
            throw std::system_error(errno, std::generic_category(), "open");

        if (fstat(fd, &sb) == -1) /* To obtain file size */
            throw std::system_error(errno, std::generic_category(), "fstat");

        size = sb.st_size;
        void *addrmm = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
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
        memcpy(addr, addrmm, size);

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

    BinFile::BinFile(std::string fileName, std::string _type, uint32_t maxVersion, bool _directRead)
        : addr(nullptr), size(0), pos(0), directRead(_directRead), fileFd(-1), version(0), readingSection(nullptr)
    {
        if (!_directRead) {
           throw std::invalid_argument("This constructor is only for direct read mode right now.");
        }

        fileFd = open(fileName.c_str(), O_RDONLY);
        if (fileFd == -1)
            throw std::system_error(errno, std::generic_category(), "open");

        struct stat sb;
        if (fstat(fileFd, &sb) == -1) {
            close(fileFd);
            fileFd = -1;
            throw std::system_error(errno, std::generic_category(), "fstat");
        }

        size = sb.st_size;

        // Read the fixed-size file header: 4 bytes type + 4 bytes version + 4 bytes nSections = 12 bytes
        uint8_t fileHeader[12];
        ssize_t bytesRead = ::pread(fileFd, fileHeader, 12, 0);
        if (bytesRead < 12) {
            close(fileFd);
            fileFd = -1;
            throw std::runtime_error("Failed to read BinFile header");
        }

        type.assign((const char *)fileHeader, 4);
        if (type != _type) {
            close(fileFd);
            fileFd = -1;
            throw std::invalid_argument("Invalid file type. It should be " + _type + " and it us " + type);
        }

        version = *((u_int32_t *)(fileHeader + 4));
        if (version > maxVersion) {
            close(fileFd);
            fileFd = -1;
            throw std::invalid_argument("Invalid version. It should be <=" + std::to_string(maxVersion) + " and it us " + std::to_string(version));
        }

        u_int32_t nSections = *((u_int32_t *)(fileHeader + 8));

        
        u_int64_t filePos = 12; // after type + version + nSections
        uint8_t entryBuf[12];
        for (u_int32_t i = 0; i < nSections; i++) {
            ssize_t r = ::pread(fileFd, entryBuf, 12, filePos);
            if (r < 12) {
                close(fileFd);
                fileFd = -1;
                throw std::runtime_error("Failed to read section entry " + std::to_string(i));
            }
            u_int32_t sType = *((u_int32_t *)(entryBuf));
            u_int64_t sSize = *((u_int64_t *)(entryBuf + 4));
            filePos += 12; // skip past the entry header

            if (sections.find(sType) == sections.end()) {
                sections.insert(std::make_pair(sType, std::vector<Section>()));
            }

            // In directRead mode, Section.start stores the FILE OFFSET (cast to void*)
            sections[sType].push_back(Section((void *)filePos, sSize));

            filePos += sSize; // skip past the section data

            if (filePos > size) {
                close(fileFd);
                fileFd = -1;
                throw std::runtime_error("Section data exceeds file size");
            }
        }

        pos = 0;
        readingSection = nullptr;
    }

    BinFile::~BinFile()
    {
        if (directRead) {
            if (fileFd >= 0) close(fileFd);
        } else {
            free(addr);
        }
    }

    void BinFile::readSectionTo(void *dest, u_int32_t sectionId, u_int64_t offset, u_int64_t len)
    {
        if (!directRead) {
            // Eager mode: copy from in-memory buffer
            void *src = (void *)((u_int64_t)getSectionData(sectionId) + offset);
            memcpy(dest, src, len);
            return;
        }

        if (sections.find(sectionId) == sections.end()) {
            throw std::range_error("Section does not exist: " + std::to_string(sectionId));
        }

        // In directRead mode, Section.start is the file offset
        u_int64_t fileOffset = (u_int64_t)sections[sectionId][0].start + offset;
        u_int64_t sectionSize = sections[sectionId][0].size;

        if (offset + len > sectionSize) {
            throw std::range_error("readSectionTo: offset+len exceeds section size");
        }

        // Read in a loop to handle partial reads
        u_int64_t totalRead = 0;
        while (totalRead < len) {
            ssize_t r = ::pread(fileFd, (uint8_t *)dest + totalRead, len - totalRead, fileOffset + totalRead);
            if (r <= 0) {
                throw std::system_error(errno, std::generic_category(), "pread in readSectionTo");
            }
            totalRead += r;
        }
    }

    void BinFile::readSectionToParallel(void *dest, u_int32_t sectionId,
                                         u_int64_t offset, u_int64_t len, int numThreads)
    {
        if (!directRead || len == 0) {
            readSectionTo(dest, sectionId, offset, len);
            return;
        }

        if (sections.find(sectionId) == sections.end()) {
            throw std::range_error("Section does not exist: " + std::to_string(sectionId));
        }

        u_int64_t fileOffset = (u_int64_t)sections[sectionId][0].start + offset;
        u_int64_t sectionSize = sections[sectionId][0].size;

        if (offset + len > sectionSize) {
            throw std::range_error("readSectionToParallel: offset+len exceeds section size");
        }

        u_int64_t chunkSize = (len + numThreads - 1) / numThreads;
        std::vector<std::thread> threads;
        for (int i = 0; i < numThreads; i++) {
            u_int64_t off = (u_int64_t)i * chunkSize;
            if (off >= len) break;
            u_int64_t sz = std::min(chunkSize, len - off);
            threads.emplace_back([fd = this->fileFd, dest, fileOffset, off, sz]() {
                u_int64_t done = 0;
                while (done < sz) {
                    ssize_t r = ::pread(fd, (uint8_t *)dest + off + done,
                                        sz - done, fileOffset + off + done);
                    if (r <= 0) {
                        throw std::system_error(errno, std::generic_category(), "pread parallel");
                    }
                    done += r;
                }
            });
        }
        for (auto &t : threads) t.join();
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
        if (directRead) {
            throw std::runtime_error("Direct read mode not supported for getSectionData");
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