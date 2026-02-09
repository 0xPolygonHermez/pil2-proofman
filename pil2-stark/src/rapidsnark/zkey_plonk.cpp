#include <stdexcept>
#include <memory.h>

#include "zkey.hpp"
#include "zkey_plonk.hpp"

namespace Zkey {
    PlonkZkeyHeader::PlonkZkeyHeader() {
        this->protocolId = Zkey::PLONK_PROTOCOL_ID;
    }

    PlonkZkeyHeader::~PlonkZkeyHeader() {
        mpz_clear(qPrime);
        mpz_clear(rPrime);
    }

    PlonkZkeyHeader* PlonkZkeyHeader::loadPlonkZkeyHeader(BinFileUtils::BinFile *f) {
        auto plonkZkeyHeader = new PlonkZkeyHeader();

        f->startReadSection(Zkey::ZKEY_PL_HEADER_SECTION);

        plonkZkeyHeader->n8q = f->readU32LE();
        mpz_init(plonkZkeyHeader->qPrime);
        mpz_import(plonkZkeyHeader->qPrime, plonkZkeyHeader->n8q, -1, 1, -1, 0, f->read(plonkZkeyHeader->n8q));

        plonkZkeyHeader->n8r = f->readU32LE();
        mpz_init(plonkZkeyHeader->rPrime);
        mpz_import(plonkZkeyHeader->rPrime, plonkZkeyHeader->n8r, -1, 1, -1, 0, f->read(plonkZkeyHeader->n8r));

        plonkZkeyHeader->nVars = f->readU32LE();
        plonkZkeyHeader->nPublic = f->readU32LE();
        plonkZkeyHeader->domainSize = f->readU32LE();
        plonkZkeyHeader->nAdditions = f->readU32LE();
        plonkZkeyHeader->nConstraints = f->readU32LE();

        plonkZkeyHeader->k1 = f->read(plonkZkeyHeader->n8r);
        plonkZkeyHeader->k2 = f->read(plonkZkeyHeader->n8r);

        plonkZkeyHeader->QM = f->read(plonkZkeyHeader->n8q * 2);
        plonkZkeyHeader->QL = f->read(plonkZkeyHeader->n8q * 2);
        plonkZkeyHeader->QR = f->read(plonkZkeyHeader->n8q * 2);
        plonkZkeyHeader->QO = f->read(plonkZkeyHeader->n8q * 2);
        plonkZkeyHeader->QC = f->read(plonkZkeyHeader->n8q * 2);

        plonkZkeyHeader->S1 = f->read(plonkZkeyHeader->n8q * 2);
        plonkZkeyHeader->S2 = f->read(plonkZkeyHeader->n8q * 2);
        plonkZkeyHeader->S3 = f->read(plonkZkeyHeader->n8q * 2);

        plonkZkeyHeader->X2 = f->read(plonkZkeyHeader->n8q * 4);

        f->endReadSection();

        return plonkZkeyHeader;
    }

    PlonkZkeyHeader* PlonkZkeyHeader::loadPlonkZkeyHeaderDirect(BinFileUtils::BinFile *f) {
        auto plonkZkeyHeader = new PlonkZkeyHeader();

        // Read entire header section into a local buffer 
        u_int64_t sectionSize = f->getSectionSize(Zkey::ZKEY_PL_HEADER_SECTION);
        uint8_t *buf = new uint8_t[sectionSize];
        f->readSectionTo(buf, Zkey::ZKEY_PL_HEADER_SECTION, 0, sectionSize);

        u_int64_t off = 0;

        plonkZkeyHeader->n8q = *((u_int32_t *)(buf + off));
        off += 4;
        mpz_init(plonkZkeyHeader->qPrime);
        mpz_import(plonkZkeyHeader->qPrime, plonkZkeyHeader->n8q, -1, 1, -1, 0, buf + off);
        off += plonkZkeyHeader->n8q;

        plonkZkeyHeader->n8r = *((u_int32_t *)(buf + off));
        off += 4;
        mpz_init(plonkZkeyHeader->rPrime);
        mpz_import(plonkZkeyHeader->rPrime, plonkZkeyHeader->n8r, -1, 1, -1, 0, buf + off);
        off += plonkZkeyHeader->n8r;

        plonkZkeyHeader->nVars = *((u_int32_t *)(buf + off));
        off += 4;
        plonkZkeyHeader->nPublic = *((u_int32_t *)(buf + off));
        off += 4;
        plonkZkeyHeader->domainSize = *((u_int32_t *)(buf + off));
        off += 4;
        plonkZkeyHeader->nAdditions = *((u_int32_t *)(buf + off));
        off += 4;
        plonkZkeyHeader->nConstraints = *((u_int32_t *)(buf + off));
        off += 4;

        // Deep-copy small pointer fields — they own their memory
        auto copyField = [](uint8_t *src, size_t len) -> void* {
            void *copy = malloc(len);
            memcpy(copy, src, len);
            return copy;
        };

        plonkZkeyHeader->k1 = copyField(buf + off, plonkZkeyHeader->n8r);
        off += plonkZkeyHeader->n8r;
        plonkZkeyHeader->k2 = copyField(buf + off, plonkZkeyHeader->n8r);
        off += plonkZkeyHeader->n8r;

        plonkZkeyHeader->QM = copyField(buf + off, plonkZkeyHeader->n8q * 2);
        off += plonkZkeyHeader->n8q * 2;
        plonkZkeyHeader->QL = copyField(buf + off, plonkZkeyHeader->n8q * 2);
        off += plonkZkeyHeader->n8q * 2;
        plonkZkeyHeader->QR = copyField(buf + off, plonkZkeyHeader->n8q * 2);
        off += plonkZkeyHeader->n8q * 2;
        plonkZkeyHeader->QO = copyField(buf + off, plonkZkeyHeader->n8q * 2);
        off += plonkZkeyHeader->n8q * 2;
        plonkZkeyHeader->QC = copyField(buf + off, plonkZkeyHeader->n8q * 2);
        off += plonkZkeyHeader->n8q * 2;

        plonkZkeyHeader->S1 = copyField(buf + off, plonkZkeyHeader->n8q * 2);
        off += plonkZkeyHeader->n8q * 2;
        plonkZkeyHeader->S2 = copyField(buf + off, plonkZkeyHeader->n8q * 2);
        off += plonkZkeyHeader->n8q * 2;
        plonkZkeyHeader->S3 = copyField(buf + off, plonkZkeyHeader->n8q * 2);
        off += plonkZkeyHeader->n8q * 2;

        plonkZkeyHeader->X2 = copyField(buf + off, plonkZkeyHeader->n8q * 4);
        off += plonkZkeyHeader->n8q * 4;

        delete[] buf;

        return plonkZkeyHeader;
    }
}
