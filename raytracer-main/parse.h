#ifndef PARSER_H
#define PARSER_H

#include <vector>
#include <string>
#include "raycast.h"  // Include struct definitions

// Function to parse a file and populate scene objects
bool isWhitespaceOnly(const std::string& str);
void parse(const std::string& filename, Scene& scene) ;
#endif // PARSER_H
