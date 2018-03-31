#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <array>
#include <random>
#include <list>
#include <chrono>
#include <omp.h>
#include <limits>
#include <algorithm>
#include <map>
#include <queue>
#include <thread>
#include <csignal>
using namespace std;
using namespace std::chrono;

constexpr int W{3};
constexpr int N{2};//Number of players
constexpr bool Debug_AI{false},Timeout{false};
constexpr int PIPE_READ{0},PIPE_WRITE{1};
constexpr double FirstTurnTime{1*(Timeout?1:10)},TimeLimit{0.1*(Timeout?1:10)};

bool stop{false};//Global flag to stop all arena threads when SIGTERM is received

struct vec{
    int x,y;
    inline bool operator==(const vec &a)const{
        return x==a.x && y==a.y;
    }
    inline bool operator!=(const vec &a)const{
        return x!=a.x || y!=a.y;
    }
    inline constexpr int idx()const{
        return y*W+x;
    }
    inline vec sub_board()const{
        return vec{x/W,y/W};
    }
    inline vec sub_board_vec()const{
        return vec{x%W,y%W};
    }
    inline vec operator*(const int a)const{
        return vec{x*a,y*a};
    }
    inline vec operator+(const vec &a)const{
        return vec{x+a.x,y+a.y};
    }
};

inline istream& operator>>(istream &is,vec &r){
    is >> r.y >> r.x;
    return is;
}

inline ostream& operator<<(ostream &os,const vec &r){
    os << r.y << " " << r.x;
    return os;
}

struct mini_board{
    private:
        array<int,W*W> G;
        double board_eval;
        inline int complete()const{
            for(int y=0;y<W;++y){
                if(G[0+y*W]!=0 && G[0+y*W]==G[1+y*W] && G[1+y*W]==G[2+y*W]){
                    return G[0+y*W];
                }
            }
            for(int x=0;x<W;++x){
                if(G[0*W+x]!=0 && G[0*W+x]==G[1*W+x] && G[1*W+x]==G[2*W+x]){
                    return G[0*W+x];
                }
            }
            if(G[0*W+0]!=0 && G[0*W+0]==G[1*W+1] && G[1*W+1]==G[2*W+2]){
                return G[0*W+0];
            }
            if(G[2*W+0]!=0 && G[2*W+0]==G[1*W+1] && G[1*W+1]==G[0*W+2]){
                return G[2*W+0];
            }
            return 0;
        }
    public:
        int winner;
        inline void reset(){
            fill(G.begin(),G.end(),0);
            winner=0;
        }
        inline int& operator[](const int &idx){
            return G[idx];
        }
        inline const int& operator[](const int &idx)const{
            return G[idx];
        }
        inline int& operator[](const vec &r){
            return G[r.idx()];
        }
        inline const int& operator[](const vec &r)const{
            return G[r.idx()];
        }
        inline void update_winner(){
            winner=complete();
        }
        inline bool full()const{
            return none_of(G.begin(),G.end(),[](const int a){return a==0;});
        }
};

struct state{
    private:
        array<mini_board,W*W> G;
        vec current;
        int winner;
        inline int complete()const{
            for(int y=0;y<W;++y){
                if(G[0+y*W].winner!=0 && G[0+y*W].winner==G[1+y*W].winner && G[1+y*W].winner==G[2+y*W].winner){
                    return G[0+y*W].winner;
                }
            }
            for(int x=0;x<W;++x){
                if(G[0*W+x].winner!=0 && G[0*W+x].winner==G[1*W+x].winner && G[1*W+x].winner==G[2*W+x].winner){
                    return G[0*W+x].winner;
                }
            }
            if(G[0*W+0].winner!=0 && G[0*W+0].winner==G[1*W+1].winner && G[1*W+1].winner==G[2*W+2].winner){
                return G[0*W+0].winner;
            }
            if(G[2*W+0].winner!=0 && G[2*W+0].winner==G[1*W+1].winner && G[1*W+1].winner==G[0*W+2].winner){
                return G[2*W+0].winner;
            }
            return 0;
        }
        inline void update_winner(){
            winner=complete();
        }
    public:
        inline void reset(){
            for_each(G.begin(),G.end(),[](mini_board &b){b.reset();});
            current=vec{W*W+1,W*W};
            winner=0;
        }
        inline void set_current(const vec &r){
            current=r;
        }
        inline vec get_current()const{
            return current;
        }
        inline bool unrestricted_turn()const{
            return current==vec{W*W+1,W*W};
        }
        inline bool board_finished(const vec &b)const{
            return G[b.idx()].winner!=0;
        }
        inline const int& operator[](const vec &r)const{
            return G[r.sub_board().idx()][r.sub_board_vec()];
        }
        inline const mini_board& get_mini_board(const vec &r)const{
            return G[r.idx()];
        }
        inline const mini_board& get_current_board()const{
            return G[current.idx()];
        }
        inline void mark(const vec &r,const int id){
            const vec mini=r.sub_board();
            if(G[mini.idx()][r.sub_board_vec()]!=0 || (!unrestricted_turn() && mini!=current)){
                if(G[mini.idx()][r.sub_board_vec()]!=0){
                    cerr << r << " is already marked" << endl;
                }
                else if(!unrestricted_turn() && mini!=current){
                    cerr << r << " is on the wrong mini board: " << mini << " instead of " << current << endl;
                }
                throw(3);
            }
            G[mini.idx()][r.sub_board_vec()]=id;
            G[mini.idx()].update_winner();
            if(G[mini.idx()].winner!=0){
                update_winner();
            }
            if(G[r.sub_board_vec().idx()].winner!=0 || G[r.sub_board_vec().idx()].full()){
                current=vec{W*W+1,W*W};
            }
            else{
                current=r.sub_board_vec();
            }
        }
        inline int get_winner()const{
            return winner;
        }
        inline double TieBreak()const{
            const int board_diff{accumulate(G.begin(),G.end(),0,[&](const int total,const mini_board &b){return total+b.winner;})};
            if(board_diff!=0){
                return board_diff>0?numeric_limits<double>::max():-numeric_limits<double>::max();
            }
            else{
                return 0;
            }
        }
        int Valid_Moves()const{
            int moves{0};
            if(unrestricted_turn()){
                for(int g=0;g<9;++g){
                    const vec b{g%W,g/W};
                    if(!board_finished(b)){
                        const mini_board& board=get_mini_board(b);
                        for(int y=0;y<W;++y){
                            for(int x=0;x<W;++x){
                                const vec r{x,y};
                                if(board[r]==0){
                                    ++moves;
                                }
                            }
                        }   
                    }
                }
            }
            else{
                const mini_board& board=get_current_board();
                for(int y=0;y<W;++y){
                    for(int x=0;x<W;++x){
                        const vec r{x,y};
                        if(board[r]==0){
                            ++moves;
                        }
                    }
                }
            }
            return moves;
        }
};

typedef vec action;

inline string EmptyPipe(const int fd){
    int nbytes;
    if(ioctl(fd,FIONREAD,&nbytes)<0){
        throw(4);
    }
    string out;
    out.resize(nbytes);
    if(read(fd,&out[0],nbytes)<0){
        throw(4);
    }
    return out;
}

struct AI{
    int id,pid,outPipe,errPipe,inPipe,turnOfDeath;
    string name;
    inline void stop(const int turn=-1){
        if(alive()){
            kill(pid,SIGTERM);
            int status;
            waitpid(pid,&status,0);//It is necessary to read the exit code for the process to stop
            if(!WIFEXITED(status)){//If not exited normally try to "kill -9" the process
                kill(pid,SIGKILL);
            }
            turnOfDeath=turn;
        }
    }
    inline bool alive()const{
        return kill(pid,0)!=-1;//Check if process is still running
    }
    inline void Feed_Inputs(const string &inputs){
        if(write(inPipe,&inputs[0],inputs.size())!=inputs.size()){
            throw(5);
        }
    }
    inline ~AI(){
        close(errPipe);
        close(outPipe);
        close(inPipe);
        stop();
    }
};

void StartProcess(AI &Bot){
    int StdinPipe[2];
    int StdoutPipe[2];
    int StderrPipe[2];
    if(pipe(StdinPipe)<0){
        perror("allocating pipe for child input redirect");
    }
    if(pipe(StdoutPipe)<0){
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        perror("allocating pipe for child output redirect");
    }
    if(pipe(StderrPipe)<0){
        close(StderrPipe[PIPE_READ]);
        close(StderrPipe[PIPE_WRITE]);
        perror("allocating pipe for child stderr redirect failed");
    }
    int nchild{fork()};
    if(nchild==0){//Child process
        if(dup2(StdinPipe[PIPE_READ],STDIN_FILENO)==-1){// redirect stdin
            perror("redirecting stdin");
            return;
        }
        if(dup2(StdoutPipe[PIPE_WRITE],STDOUT_FILENO)==-1){// redirect stdout
            perror("redirecting stdout");
            return;
        }
        if(dup2(StderrPipe[PIPE_WRITE],STDERR_FILENO)==-1){// redirect stderr
            perror("redirecting stderr");
            return;
        }
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        close(StdoutPipe[PIPE_READ]);
        close(StdoutPipe[PIPE_WRITE]);
        close(StderrPipe[PIPE_READ]);
        close(StderrPipe[PIPE_WRITE]);
        execl(Bot.name.c_str(),Bot.name.c_str(),(char*)NULL);//(char*)Null is really important
        //If you get past the previous line its an error
        perror("exec of the child process");
    }
    else if(nchild>0){//Parent process
        close(StdinPipe[PIPE_READ]);//Parent does not read from stdin of child
        close(StdoutPipe[PIPE_WRITE]);//Parent does not write to stdout of child
        close(StderrPipe[PIPE_WRITE]);//Parent does not write to stderr of child
        Bot.inPipe=StdinPipe[PIPE_WRITE];
        Bot.outPipe=StdoutPipe[PIPE_READ];
        Bot.errPipe=StderrPipe[PIPE_READ];
        Bot.pid=nchild;
    }
    else{//failed to create child
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        close(StdoutPipe[PIPE_READ]);
        close(StdoutPipe[PIPE_WRITE]);
        perror("Failed to create child process");
    }
}

inline bool IsValidMove(const state &S,const AI &Bot,const string &M){
    return count(M.begin(),M.end(),'\n')==1;
}

string GetMove(const state &S,AI &Bot,const int turn){
    pollfd outpoll{Bot.outPipe,POLLIN};
    time_point<system_clock> Start_Time{system_clock::now()};
    string out;
    while(static_cast<duration<double>>(system_clock::now()-Start_Time).count()<(turn==1?FirstTurnTime:TimeLimit) && !IsValidMove(S,Bot,out)){
        double TimeLeft{(turn==1?FirstTurnTime:TimeLimit)-static_cast<duration<double>>(system_clock::now()-Start_Time).count()};
        if(poll(&outpoll,1,TimeLeft)){
            out+=EmptyPipe(Bot.outPipe);
        }
    }
    return out;
}

inline bool Has_Won(const array<AI,N> &Bot,const int idx)noexcept{
    if(!Bot[idx].alive()){
        return false;
    }
    for(int i=0;i<N;++i){
        if(i!=idx && Bot[i].alive()){
            return false;
        }
    }
    return true;
}

inline bool All_Dead(const array<AI,N> &Bot)noexcept{
    for(const AI &b:Bot){
        if(b.alive()){
            return false;
        }
    }
    return true;
}

action StringToAction(const state &S,const string &M_Str){
    action mv;
    stringstream ss(M_Str);
    if(ss >> mv){
        return mv;
    }
    cerr << M_Str << " is not a valid move" << endl;
    throw(3);
}

void Simulate(state &S,const action &mv,const int playerId){
    S.mark(mv,playerId==0?1:-1);
}

int Play_Game(const array<string,N> &Bot_Names,state S){
    array<AI,N> Bot;
    for(int i=0;i<N;++i){
        Bot[i].id=i;
        Bot[i].name=Bot_Names[i];
        StartProcess(Bot[i]);
    }
    int turn{0};
    action last{-1,-1};
    while(++turn>0 && !stop){
        //cerr << turn << endl;
        for(int id=0;id<N;++id){
            if(Bot[id].alive()){
                stringstream ss;
                ss << last << endl;
                ss << 0 << endl;
                //cerr << ss.str();
                try{
                    if(S.Valid_Moves()==0){
                        throw(0);
                    }
                    Bot[id].Feed_Inputs(ss.str());
                    string out=GetMove(S,Bot[id],turn);
                    //cerr << id << " " << out << endl;
                    string err_str{EmptyPipe(Bot[id].errPipe)};
                    if(Debug_AI){
                        ofstream err_out("log.txt",ios::app);
                        err_out << err_str << endl;
                    }
                    const action mv=StringToAction(S,out);
                    Simulate(S,mv,id);
                    last=mv;
                    if(S.get_winner()!=0){
                        return id==0?1:-1;
                    }
                }
                catch(int ex){
                    if(ex==1){//Timeout
                        cerr << "Loss by Timeout of AI " << Bot[id].id << " name: " << Bot[id].name << endl;
                    }
                    else if(ex==3){
                        cerr << "Invalid move from AI " << Bot[id].id << " name: " << Bot[id].name << endl;
                    }
                    else if(ex==4){
                        cerr << "Error emptying pipe of AI " << Bot[id].name << endl;
                    }
                    else if(ex==5){
                        cerr << "AI " << Bot[id].name << " died before being able to give it inputs" << endl;
                    }
                    Bot[id].stop(turn);
                }
            }
        }
        if(All_Dead(Bot)){
            const double tie{S.TieBreak()};
            return tie==numeric_limits<double>::max()?1:tie==-numeric_limits<double>::max()?-1:0;
        }
    }
    throw(0);
}

inline double WinnerToPoint(const int w){
    if(w==0){
        //cerr << "Draw" << endl;
    }
    return w==0?0.5:w==1?1:0;
}

double Play_Round(array<string,N> Bot_Names){
    state S;
    S.reset();
    array<int,2> winner;
    winner[0]=Play_Game(Bot_Names,S);
    swap(Bot_Names[0],Bot_Names[1]);
    winner[1]=-Play_Game(Bot_Names,S);
    return accumulate(winner.begin(),winner.end(),0.0,[](const double total,const int w){return total+WinnerToPoint(w);});
}

void StopArena(const int signum){
    stop=true;
}

int main(int argc,char **argv){
    if(argc<3){
        cerr << "Program takes 2 inputs, the names of the AIs fighting each other" << endl;
        return 0;
    }
    int N_Threads{1};
    if(argc>=4){//Optional N_Threads parameter
        N_Threads=min(2*omp_get_num_procs(),max(1,atoi(argv[3])));
        cerr << "Running " << N_Threads << " arena threads" << endl;
    }
    array<string,N> Bot_Names;
    for(int i=0;i<2;++i){
        Bot_Names[i]=argv[i+1];
    }
    cout << "Testing AI " << Bot_Names[0];
    for(int i=1;i<N;++i){
        cerr << " vs " << Bot_Names[i];
    }
    cerr << endl;
    for(int i=0;i<N;++i){//Check that AI binaries are present
        ifstream Test{Bot_Names[i].c_str()};
        if(!Test){
            cerr << Bot_Names[i] << " couldn't be found" << endl;
            return 0;
        }
        Test.close();
    }
    signal(SIGTERM,StopArena);//Register SIGTERM signal handler so the arena can cleanup when you kill it
    signal(SIGPIPE,SIG_IGN);//Ignore SIGPIPE to avoid the arena crashing when an AI crashes
    int draws{0},games{0};
    array<double,2> points{0,0};
    #pragma omp parallel num_threads(N_Threads) shared(points,Bot_Names,games)
    while(!stop){
        double winner{Play_Round(Bot_Names)};
        #pragma omp atomic
        points[0]+=winner;
        #pragma omp atomic
        games+=2;
        double p{points[0]/games};
        double sigma{sqrt(p*(1-p)/games)};
        double better{0.5+0.5*erf((p-0.5)/(sqrt(2)*sigma))};
        #pragma omp critical
        cout << "Wins:" << setprecision(4) << 100*p << "+-" << 100*sigma << "% Rounds:" << games << " Draws:" << draws << " " << better*100 << "% chance that " << Bot_Names[0] << " is better" << endl;
    }
}