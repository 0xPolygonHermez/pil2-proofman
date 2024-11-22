#include <sys/stat.h>
#include "expressions_bin.hpp"

#define BINFILE_VERSION "0.1.0.0"

using namespace std;

class ArgumentParser {
private:
    vector <string> arguments;
public:
    ArgumentParser (int &argc, char **argv)
    {
        for (int i=1; i < argc; ++i)
            arguments.push_back(string(argv[i]));
    }

    string getArgumentValue (const string argshort, const string arglong) 
    {
        for (size_t i=0; i<arguments.size(); i++) {
            if (argshort==arguments[i] || arglong==arguments[i]) {
                if (i+1 < arguments.size()) return (arguments[i+1]);
                else return "";
            }
        }
        return "";
    }

    bool argumentExists (const string argshort, const string arglong) 
    {
        bool found = false;
        for (size_t i=0; i<arguments.size(); i++) {
            if (argshort==arguments[i] || arglong==arguments[i]) {
                if (found) {
                    throw runtime_error("binfile: cannot use "+argshort+"/"+arglong+" parameter twice!");
                } else found = true;
            }
        }
        return found;
    }
};

void showVersion() 
{
    cout << "binfile: version " << string(BINFILE_VERSION) << endl;
}

int main(int argc, char **argv)
{
    string starkInfoFile = "";
    string expressionsInfoFile = "";
    string expressionsBinFile = "";
    bool isGlobal = false;

    ArgumentParser aParser (argc, argv);

    try {

        isGlobal = aParser.argumentExists("-g", "--global");

        //Input arguments
        if(!isGlobal) {
            if (aParser.argumentExists("-s","--stark")) {
                starkInfoFile = aParser.getArgumentValue("-s", "--stark");
                if (!fileExists(starkInfoFile)) throw runtime_error("binfile: starkinfo file doesn't exist ("+starkInfoFile+")");
            } else throw runtime_error("binfile: starkinfo input file argument not specified <-s/--stark> <starkinfo_file>");
        }
        if (aParser.argumentExists("-e","--expsinfo")) {
            expressionsInfoFile = aParser.getArgumentValue("-e", "--expsinfo");
            if (!fileExists(expressionsInfoFile)) throw runtime_error("binfile: expressions info file doesn't exist ("+expressionsInfoFile+")");
        } else throw runtime_error("binfile: expressions info input file argument not specified <-e/--expsinfo> <expressionsinfo_file>");
        //Output arguments
        if (aParser.argumentExists("-b","--binfile")) {
            expressionsBinFile = aParser.getArgumentValue("-b","--binfile");
            if (expressionsBinFile=="") throw runtime_error("binfile: bin ouput file not specified");
        } else throw runtime_error("binfile: expressions bin file argument not specified <-b/--binfile> <bin_file>");
        
        showVersion();

        ExpressionsBin(starkInfoFile, expressionsInfoFile, expressionsBinFile, isGlobal);
        
        cout << "File successfully written" << endl;

        return EXIT_SUCCESS;
    } catch (const exception &e) {
        cerr << e.what() << endl;
        showVersion();
        cerr << "usage: binfile <-s|--stark> <starkinfo_file> <-e|--expsinfo> <expressionsinfo_file> <-b|--binfile> <bin_file> [-g|--global]" << endl;
        cerr << "example 1: binfile -s test.starkinfo.json -e test.expressionsinfo.json -b test.expressions.bin" << endl;
        cerr << "example 2: binfile -g -e global.expressionsinfo.json -b global.expressions.bin" << endl;
        return EXIT_FAILURE;        
    }    
}
