// Tiny JSON reader, enough for assets/meta.json (objects, arrays, numbers, strings, bool, null).
// Not a general-purpose library; no unicode escapes beyond the basics. Header-only.
#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdlib>

namespace fohn {

struct Json {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
  double num = 0;
  bool boolean = false;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  bool has(const std::string& k) const { return type == Obj && obj.count(k); }
  const Json& operator[](const std::string& k) const { static Json nil; auto it = obj.find(k); return it == obj.end() ? nil : it->second; }
  const Json& operator[](size_t i) const { static Json nil; return i < arr.size() ? arr[i] : nil; }
  size_t size() const { return arr.size(); }
  double n(double d = 0) const { return type == Num ? num : d; }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& s) : s_(s) {}
  Json parse() { skip(); return value(); }

 private:
  const std::string& s_;
  size_t i_ = 0;

  void skip() { while (i_ < s_.size()) { char c = s_[i_]; if (c == ' ' || c == '\t' || c == '\n' || c == '\r') i_++; else break; } }
  char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }

  Json value() {
    skip();
    char c = peek();
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') { Json j; j.type = Json::Str; j.str = string(); return j; }
    if (c == 't' || c == 'f') { Json j; j.type = Json::Bool; j.boolean = (c == 't'); i_ += (c == 't') ? 4 : 5; return j; }
    if (c == 'n') { i_ += 4; return Json(); }
    return number();
  }

  Json object() {
    Json j; j.type = Json::Obj; i_++; skip();
    if (peek() == '}') { i_++; return j; }
    while (true) {
      skip();
      std::string key = string();
      skip(); if (peek() == ':') i_++;
      j.obj[key] = value();
      skip();
      char c = peek();
      if (c == ',') { i_++; continue; }
      if (c == '}') { i_++; break; }
      break;
    }
    return j;
  }

  Json array() {
    Json j; j.type = Json::Arr; i_++; skip();
    if (peek() == ']') { i_++; return j; }
    while (true) {
      j.arr.push_back(value());
      skip();
      char c = peek();
      if (c == ',') { i_++; continue; }
      if (c == ']') { i_++; break; }
      break;
    }
    return j;
  }

  std::string string() {
    std::string out;
    if (peek() == '"') i_++;
    while (i_ < s_.size()) {
      char c = s_[i_++];
      if (c == '"') break;
      if (c == '\\' && i_ < s_.size()) {
        char e = s_[i_++];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          default: out += e; break;   // skip \uXXXX handling; meta.json keys are ASCII
        }
      } else {
        out += c;
      }
    }
    return out;
  }

  Json number() {
    size_t start = i_;
    while (i_ < s_.size()) {
      char c = s_[i_];
      if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') i_++;
      else break;
    }
    Json j; j.type = Json::Num; j.num = std::strtod(s_.c_str() + start, nullptr);
    return j;
  }
};

inline Json parseJson(const std::string& s) { JsonParser p(s); return p.parse(); }

} // namespace fohn
