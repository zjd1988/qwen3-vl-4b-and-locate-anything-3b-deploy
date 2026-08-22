// locate-anything-server: HTTP frontend for la::Engine, in the spirit of llama-server.
//
// Endpoints:
//   GET  /health                 -> {"status":"ok"}
//   GET  /v1/models              -> OpenAI-style model list (single entry)
//   POST /v1/detect              -> detection JSON (same schema as the CLI)
//        multipart/form-data: image=<file>, prompt=<text>, [mode=hybrid|slow|fast], [max_new=N]
//        application/json:    {"image_base64":"...","prompt":"...","mode":"hybrid","max_new":256}
//
// The engine is stateful and not thread-safe, so all inference calls are
// serialized through a single mutex (the httplib thread pool still handles
// connection concurrency).
#include "engine.hpp"
#include "image_io.hpp"
#include "httplib.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

std::string json_escape(const std::string& s){
    std::string o; o.reserve(s.size()+8);
    for(char c : s){
        switch(c){
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if(static_cast<unsigned char>(c) < 0x20){ char b[8]; std::snprintf(b,sizeof b,"\\u%04x",c); o += b; }
                else o += c;
        }
    }
    return o;
}

std::string detections_json(const std::vector<la::Box>& b){
    std::string j = "\"detections\":["; char buf[512];
    for(size_t i=0;i<b.size();++i){ std::snprintf(buf,sizeof buf,
        "%s{\"label\":\"%s\",\"box\":[%.2f,%.2f,%.2f,%.2f]}", i?",":"",
        json_escape(b[i].label).c_str(), b[i].x1,b[i].y1,b[i].x2,b[i].y2); j+=buf; }
    j+="]"; return j;
}

std::string error_json(const std::string& msg, const std::string& type){
    return "{\"error\":{\"message\":\""+json_escape(msg)+"\",\"type\":\""+type+"\"}}";
}

// Minimal flat-object JSON helpers: enough for {"image_base64","prompt","mode","max_new"}.
// Handles standard string escapes; not a general JSON parser.
bool json_get_string(const std::string& body, const std::string& key, std::string& out){
    std::string pat = "\""+key+"\"";
    size_t p = body.find(pat); if(p==std::string::npos) return false;
    p = body.find(':', p+pat.size()); if(p==std::string::npos) return false;
    p = body.find('"', p+1); if(p==std::string::npos) return false;
    std::string v;
    for(size_t i=p+1;i<body.size();++i){
        char c = body[i];
        if(c=='\\' && i+1<body.size()){
            char e = body[++i];
            switch(e){
                case 'n': v+='\n'; break; case 't': v+='\t'; break; case 'r': v+='\r'; break;
                case 'u': // \u00XX basic BMP escape (enough for prompts)
                    if(i+4<body.size()){ int cp=0; std::sscanf(body.c_str()+i+1,"%4x",&cp); i+=4;
                        if(cp<0x80) v+=(char)cp;
                        else if(cp<0x800){ v+=(char)(0xC0|(cp>>6)); v+=(char)(0x80|(cp&0x3F)); }
                        else { v+=(char)(0xE0|(cp>>12)); v+=(char)(0x80|((cp>>6)&0x3F)); v+=(char)(0x80|(cp&0x3F)); } }
                    break;
                default: v+=e;
            }
        } else if(c=='"'){ out=v; return true; }
        else v+=c;
    }
    return false;
}

bool json_get_int(const std::string& body, const std::string& key, int& out){
    std::string pat = "\""+key+"\"";
    size_t p = body.find(pat); if(p==std::string::npos) return false;
    p = body.find(':', p+pat.size()); if(p==std::string::npos) return false;
    ++p; while(p<body.size() && (body[p]==' '||body[p]=='\t')) ++p;
    if(p>=body.size()) return false;
    char* end=nullptr; long v=std::strtol(body.c_str()+p,&end,10);
    if(end==body.c_str()+p) return false;
    out=(int)v; return true;
}

const char* b64_table(){
    static char t[256]; static bool init=false;
    if(!init){ std::memset(t,-1,sizeof t);
        for(int i=0;i<26;++i){ t['A'+i]=i; t['a'+i]=26+i; }
        for(int i=0;i<10;++i) t['0'+i]=52+i;
        t['+']=62; t['/']=63; init=true; }
    return t;
}

bool base64_decode(const std::string& in, std::vector<unsigned char>& out){
    const char* T = b64_table(); out.clear(); out.reserve(in.size()*3/4);
    int val=0, bits=-8;
    for(unsigned char c : in){
        if(c=='=' || c=='\n' || c=='\r' || c==' ') continue;
        if(T[c]<0) return false;
        val=(val<<6)+T[c]; bits+=6;
        if(bits>=0){ out.push_back((unsigned char)((val>>bits)&0xFF)); bits-=8; }
    }
    return true;
}

struct Args {
    std::string model;              // required
    std::string host = "0.0.0.0";
    int port = 8081;
    int threads = 0;                // 0 = hardware_concurrency (Engine default)
};

void print_help(){
    std::printf("locate-anything-server — HTTP server for LocateAnything-3B\n"
                "usage: locate-anything-server --model MODEL.gguf [options]\n"
                "options:\n"
                "  --model, -m PATH   model GGUF (required)\n"
                "  --host HOST        listen host (default 0.0.0.0)\n"
                "  --port PORT        listen port (default 8081)\n"
                "  --threads, -t N    CPU threads for non-offloaded work (default: all cores)\n"
                "  --help, -h         show this help\n");
}

bool parse_args(int argc, char** argv, Args& a){
    for(int i=1;i<argc;++i){
        std::string k = argv[i];
        auto next = [&](std::string& v)->bool{ if(i+1>=argc) return false; v=argv[++i]; return true; };
        std::string v;
        if(k=="--model"||k=="-m"){ if(!next(a.model)) return false; }
        else if(k=="--host"){ if(!next(a.host)) return false; }
        else if(k=="--port"){ if(!next(v)) return false; a.port=std::atoi(v.c_str()); }
        else if(k=="--threads"||k=="-t"){ if(!next(v)) return false; a.threads=std::atoi(v.c_str()); }
        else if(k=="--help"||k=="-h"){ print_help(); std::exit(0); }
        else return false;
    }
    return !a.model.empty() && a.port>0 && a.port<65536;
}

}  // namespace

int main(int argc, char** argv){
    Args args;
    if(!parse_args(argc, argv, args)){ print_help(); return 1; }

    auto eng = la::Engine::load(args.model, args.threads);
    if(!eng){ std::fprintf(stderr,"error: failed to load model %s\n", args.model.c_str()); return 1; }
    std::fprintf(stderr,"model loaded: %s\n", args.model.c_str());

    std::mutex eng_mu;
    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res){
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res){
        res.set_content("{\"object\":\"list\",\"data\":[{\"id\":\"locate-anything-3b\","
                        "\"object\":\"model\",\"owned_by\":\"locate-anything.cpp\"}]}",
                        "application/json");
    });

    auto detect = [&](const httplib::Request& req, httplib::Response& res){
        std::string image_bytes, prompt, mode_str = "hybrid";
        int max_new = 256;

        if(req.is_multipart_form_data()){
            if(!req.form.has_file("image")){ res.status=400;
                res.set_content(error_json("missing multipart field 'image'","invalid_request_error"),"application/json"); return; }
            image_bytes = req.form.get_file("image").content;
            if(req.form.has_field("prompt"))  prompt   = req.form.get_field("prompt");
            if(req.form.has_field("mode"))    mode_str = req.form.get_field("mode");
            if(req.form.has_field("max_new")) max_new  = std::atoi(req.form.get_field("max_new").c_str());
        } else {
            std::string b64;
            if(!json_get_string(req.body,"image_base64",b64)){ res.status=400;
                res.set_content(error_json("JSON body requires 'image_base64' and 'prompt' (or use multipart/form-data with an 'image' file)","invalid_request_error"),"application/json"); return; }
            std::vector<unsigned char> decoded;
            if(!base64_decode(b64, decoded) || decoded.empty()){ res.status=400;
                res.set_content(error_json("invalid base64 in 'image_base64'","invalid_request_error"),"application/json"); return; }
            image_bytes.assign((const char*)decoded.data(), decoded.size());
            json_get_string(req.body,"prompt",prompt);
            std::string m; if(json_get_string(req.body,"mode",m)) mode_str=m;
            int mn; if(json_get_int(req.body,"max_new",mn)) max_new=mn;
        }

        if(prompt.empty()){ res.status=400;
            res.set_content(error_json("missing 'prompt'","invalid_request_error"),"application/json"); return; }

        la::Engine::Mode mode;
        if(mode_str=="hybrid") mode=la::Engine::Mode::Hybrid;
        else if(mode_str=="slow") mode=la::Engine::Mode::Slow;
        else if(mode_str=="fast") mode=la::Engine::Mode::Fast;
        else { res.status=400;
            res.set_content(error_json("invalid 'mode' (expected hybrid|slow|fast)","invalid_request_error"),"application/json"); return; }
        if(max_new<=0) max_new=256;

        // Decode up front so a bad image is a 400, and to report original dims.
        la::Image img;
        if(!la::load_image_rgb_buffer((const unsigned char*)image_bytes.data(), image_bytes.size(), img)){
            res.status=400;
            res.set_content(error_json("failed to decode image (PNG/JPEG expected)","invalid_request_error"),"application/json"); return;
        }

        auto t0 = std::chrono::steady_clock::now();
        std::vector<la::Box> boxes;
        {
            std::lock_guard<std::mutex> lk(eng_mu);
            boxes = eng->locate_buffer((const unsigned char*)image_bytes.data(), image_bytes.size(),
                                       prompt, mode, max_new);
        }
        auto t1 = std::chrono::steady_clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

        char meta[1024];
        std::snprintf(meta,sizeof meta,
            ",\"prompt\":\"%s\",\"mode\":\"%s\",\"image\":{\"width\":%d,\"height\":%d},\"elapsed_ms\":%lld}",
            json_escape(prompt).c_str(), mode_str.c_str(), img.w, img.h, ms);
        res.set_content("{"+detections_json(boxes)+meta, "application/json");
    };
    svr.Post("/v1/detect", detect);
    svr.Post("/detect", detect);

    std::fprintf(stderr,"listening on http://%s:%d (endpoints: /health, /v1/models, /v1/detect)\n",
                 args.host.c_str(), args.port);
    if(!svr.listen(args.host, args.port)){
        std::fprintf(stderr,"error: failed to bind %s:%d\n", args.host.c_str(), args.port);
        return 1;
    }
    return 0;
}
