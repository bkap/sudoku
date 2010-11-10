from __future__ import print_function, division
from unittest import TestCase, main

'''This is a command-line sudoku solver. You can either enter a file or type the input into the command line.'''

class InvalidNumberException(BaseException) :
    def __init__(self, *args, **kwargs) :
        BaseException.__init__(self, args, kwargs)
        if 'cell' in kwargs :
            self.cell = kwargs['cell']
        
class Cell(object) :
    """this object represents a single square in a Sudoku. It contains a number from 1 to 9.
    If it has not yet been given a number, it contains a list of all possible values. As the board
    becomes filled in, values are removed from this list"""
    def __init__(self, number = None, max_possible = 9) :
        """a single cell in Sudoku. If it does not start with a number,
        then it's number is None and it will be given a list of possible numbers"""
        self.possibilities = [True for _ in range(max_possible)]
        self._set_number(number)
        
    @classmethod
    def copy(cls, other_cell) :
        """create a duplicate of this cell"""
        new_obj = cls.__new__(cls)
        new_obj.__init__(number = other_cell.number, max_possible = len(other_cell.possibilities))
        new_obj.possibilities = other_cell.possibilities[:]
        return new_obj
   
    #Descripter for number
    def _set_number(self, number = None) :
        if number is not None and number > len(self.possibilities) :
            raise InvalidNumberException("That number is too high for this grid")
        if number is not None :
            if not self.possibilities[number - 1] :
                raise InvalidNumberException("That number is already used",cell = number)
            else :
                self.possibilities = [(False if i != number - 1 else True) for i in range(len(self.possibilities))]
                if not _in_init :
                    pass
        self._number = number
        
    def _get_number(self) :
        return self._number
    
    
    def __str__(self) :
        if self.number is not None :
            return "<Cell object, num= {0}>".format(self.number)
        else :
            return "<Cell object, possibilities = {0}>".format(self.possibilities)
    __repr__ = __str__
    
 
    
    number = property(_get_number, _set_number, doc='''set's the number of this cell. raises InvalidNumberException if the number is less than 1
                                                        or greater than the number of cells in the row''')
    
    def get_possibilities(self) :
        x = set([x if poss else None for (x,poss) in enumerate(self.possibilities)])
        if None in x :
            x.remove(None)
        return x
    
    def remove_possibility(self, number) :
        """says that the given number is no longer a possiblity for this cell. Then checks to
        see if the cell's value is now known"""
        self.possibilities[number - 1] = False
        return self.check_for_one()
    
    def is_possibility(self, number) :
        """checks to see if the number is still a potential value for this cell"""
        return self.possibilities[number - 1]

    def check_for_one(self) :
        '''check to see if there is only one remaining possibility for this cell. If there is, set the cell '''
        if self.possibilities.count(True) == 1 :
            self.number = self.possibilities.index(True) + 1
            return True
        return False
    
    
class InvalidSolutionException(BaseException) :
    '''This is an Exception thrown if the sudoku is not solvable. Used to signal to the Sudoku Solver that it should revert and try again'''
    def __init__(self, *args, **kwargs):
        BaseException.__init__(self, args, kwargs)
        self.cell_num = kwargs['cell']

class Region(object) :
    """this defines a region in Sudoku. A region is either a column, row, or nxn box. In a region, every number
    from 1 to len(cells) must appear exactly once"""
    def __init__(self, cells) :
        self.cells = cells
 
                
    def __str__(self) :
        return str(self.cells)
    
    def fix_cells(self, guess = False) :
        '''attempt to eliminate some possibilities in some of the cells. If guess is True, 
            the solver will make a best guess as to a possible solution. Otherwise, it will only eliminate
            things that can be logically eliminated'''
        self.numbers = []
        num_set = False
        for i,cell in enumerate(self.cells) :
            if cell.number is not None :
                if cell.number in self.numbers :
                    raise InvalidSolutionException(cell=i)
                self.numbers.append(cell.number)
        for cell in self.cells :
            if cell.number is None :
                this_set = False
                for number in self.numbers :
                    this_set |= cell.remove_possibility(number)
                    if this_set :
                        num_set = True
                        self.numbers.append(cell.number)
                        self.fix_cells()
                        break
        num_set |= self.check_if_only_one()
        if guess:
            num_set |= self.find_a_guess()
        if num_set :
            self.fix_cells()
        return num_set
    
    def find_a_guess(self) :
        for i,cell in enumerate(self.cells) :
            if len(cell.get_possibilities()) == 2:
                #try the first one
                cell.number = sorted(cell.get_possibilities())[0] + 1
                self.guess = i
                return True
        return False

    def check_if_only_one(self) :
        """this should only be called during solve. It checks to see if there
        is only one cell in the grid that can be a particular number. For instance, the * in the example.
        That number can be set. Obviously, the real case would have to be harder
        to justify using this technique, but it would work.
        
        | | | | | | | | |2|
        |1|*|3|4|5|6|7|8| |
        """
        to_check = self.cells[:]
        i = 0
        num_set = False
        while i < len(to_check) :
            if to_check[i].number is not None :
                del to_check[i]
            else :
                i += 1
        for i in range(1,len(self.cells) + 1) :
            possible_i = [x for x in to_check if x.is_possibility(i)]
            num_of_i = len(possible_i)
            if num_of_i == 1 :
                possible_i[0].number = i
                num_set = True
        return num_set
                
    def is_done(self) :
        """checks to see if all numbers in the region are set.
        returns True if they all are not None"""
        done = True
        for cell in self.cells :
            done &= (cell.number is not None)
        return done

class Sudoku(object) :
    """This class represents a Sudoku Grid. A sudoku grid must be a square.
        Each cell in the grid is given a number from 1 to len(grid). To solve a sudoku,
        every line, column, and block of cells must contain one of each of the 
        possible numbers. A block is a group of cells formed from part of the row and part of the
        column. In a traditional 9x9 grid, each block is 3x3. This program cannot use advanced techniques
        for solving Sudoku puzzles.
    """
    def __init__(self, grid, row_block = None, col_block = None) :
        """create a Sudoku Grid. Grid must be a 2D list, any width. 
        If blocks are squares- row_block and col_block will automatically be set
        to sqrt(length). If you want something else or the grid length isn't a perfect square,
        set them"""
        global _in_init
        _in_init = True
        import math
        #if you don't explicitly give a number of rows per block,
        #it uses the sqrt of the grid length
        if row_block is None :
            self.row_block = int(math.sqrt(len(grid)))
        else :
            self.row_block = row_block
            
        #ditto for columns
        if col_block is None :
            self.col_block = int(math.sqrt(len(grid[0])))
        else :
            self.col_block = col_block
        self.row_len = len(grid)
        print("self.row_len = %d" % self.row_len)
        self.board = [[] for i in range(self.row_len)]
        
        #now that we know how long the board has to be, lets initialize all of the 
        #cells
        for i in range(self.row_len) :
            row = self.board[i]
            for a_cell in grid[i] :
                row.append(Cell(a_cell, max_possible = self.row_len))
        _in_init = False
        self.prev_states = []
        self.guess_time = False
    def get_row(self, row_num) :
        """returns a Region consisting of every cell in the given row"""
        
        return Region(self.board[row_num])
    
    def get_col(self, col_num) :
        """returns a Region containing every cell in the given column"""
        return Region([self.board[i][col_num] for i in range(self.row_len)])
    
    def get_block(self, row, col) :
        """row goes from 0 to row_len/row_block - 1.
        col goes from 0 to row_len/col_block -1
        if row_block and col_block were given. Otherwise, they
        go from 0 to sqrt(row_len) and sqrt(col_len)"""
        cells = []
        for row_num in range(row * self.row_block, (row + 1) * self.row_block) :
            row = self.board[row_num]
            for col_num in range(col * self.col_block, (col + 1) * self.col_block) :
                cells.append(row[col_num])
        return Region(cells)
    def get_grid(self) :
        '''returns the 2-dimensional array of ints that forms this board. Used only for testing purposes'''
        return [[cell.number for cell in row] for row in self.board]
    
    
    def is_finished(self) :
        """checks to see if the Sudoku grid is completed. I.E. all cells have numbers"""
        for i in range(self.row_len) :
            if not self.get_row(i).is_done() :
                return False
        return True
    
    
    def __str__(self) :
        """returns a visual representation of the sudoku board"""
        s = ""
        for i in range(self.row_len) :
            row = self.board[i]
            for j in range(self.row_len) :
                cell = row[j]
                if cell.number == None :
                    ch = "".center(3, " ")
                else :
                    ch = str(cell.number).center(3, " ")
                if j % self.col_block == 0 :
                    s += "|"
                s += ch + "|"
            if i % self.row_block == self.row_block - 1 and i != self.row_len - 1 :
                    num_chrs = 4 * self.row_len + (self.row_len //self.row_block) 
                    s += "\n" + ("-" * num_chrs)
            s += "\n"
#        for row in self.board :
#            for cell in row :
#                if cell.number == None :
#                    s += "   |"
#                else :
#                    s += " " + str(cell.number) + " |"
            s += "\n"
        return s
    def copy_board(self) :
        c = [[Cell.copy(cell) for cell in row[:]] for row in self.board[:]]
        return c
    def solve(self) :
        """tries to solve a Sudoku. It returns True when it is finished or False if it can't finish"""
        guess = False
        guessed = False
        while not self.is_finished() :
            
            try:
               
                changed = False;
                state = 1 # 1 = row, 2 = col, 3 = block. Used in case we need to guess, so we know where we left off
                if guess and not guessed :
                    #ensure that we only guess once
                    guessed = True
                    self.prev_states.append([[],self.copy_board()])
                #test the rows
                for i in range(self.row_len) :
                    row = self.get_row(i)
                    changed |= row.fix_cells(guess)
                    if changed and guess:
                        #we should only ever guess on the rows to make sure that we don't accidentally do it twice
                        self.prev_states[-1][0] = self.get_cell_num(state,i,row.guess)
                        guess = False
                state = 2
                #test the columns
                for i in range(self.row_len) :
                    re = self.get_col(i)
                    changed |= re.fix_cells()
                state = 3
                for i in range(self.col_block) :
                    for j in range(self.row_block) :
                        changed |= self.get_block(i, j).fix_cells()
                if changed and guess:
                    guess = False
                    guessed = False
                elif not changed and guess :
                    return False
                elif not changed :

                    guess = True
                    guessed = False
            except InvalidSolutionException as e:
                if not self.prev_states :
                    #we have an invalid board at the beginning
                    raise e
                prev_state = self.prev_states.pop()
                if not prev_state[0] :
                    return False
                r,c = prev_state[0]
                prev_board = prev_state[1]
                prev_board[r][c].remove_possibility(self.board[r][c].number)
                self.board = prev_board
                    
        return True
    def get_cell_num(self,state, i, cell) :
        '''figure out which cell was changed'''
        if state == 1 : #row, it's easy
            return [i,cell]
        elif state == 2: #column, also easy
            return [cell, i]
        elif state == 3: #box, it's harder. 
            #i better be row, column
            r,c = i
            row = self.row_block * r + cell % (len(self.board[0]) //self.row_block)
            col = self.col_block * c + cell // (len(self.board[0]) // self.col_block)
            return [row, col]
    def clone(self) :
        return Sudoku(self.get_grid(), self.row_block, self.col_block)
        
def make_puzzle(f) :
    if isinstance(f, file) :
        f = f.readlines()
    elif isinstance(f, str) :
        f = f.split('\n')
    board = []
    for line in f :
            if line :
                board.append([int(x) if x.strip() else None for x in line.rstrip('\n')])

    s = Sudoku(board)
    return s

solvable = [[6,3,8,None,None,None,None, 9, None],
            [None, None, 9, 4,6,None, None, None, 2],
            [None, None, 1,7,None, 8, None, None, None],
            [None, None, 4, None, None, 1, 2, None, 7],
            [None, 6, None, 2, None, 5, None, 4, None],
            [2, None, 3, 9, None, None, 6, None, None],
            [None, None, None, 5, None, 9, 4, None, None],
            [9, None, None, None, 4,7,8,None, None],
            [None, 4,None,None,None,None,5,3,9]]
medium = [[None, 1, None, 8, None, None, 3, 6, None],
          [6,None,3,2,None,None,None,7,None],
          [None,8,None,7,None,None,None,None,9],
          [2,None,9,4,None,None,None,None,None],
          [4,None,None,None,9,None,None,None,1],
          [None,None,None,None,None,8,9,None,4],
          [9,None,None,None,None,5,None,8,None],
          [None,5,None,None,None,2,6,None,3],
          [None,2,6,None,None,4,None,9,None]]
hard = [[4,None,None,7,None,None,None,8,None],
        [None,1,3,None,None,None,None,5,None],
        [9,None,8,None,5,4,None,6,None],
        [None,None,None,None,6,None,None,None,None],
        [None,None,2,4,None,9,6,None,None],
        [None,None,None,None,7,None,None,None,None],
        [None,5,None,1,2,None,9,None,6],
        [None,6,None,None,None,None,7,2,None],
        [None,8,None,None,None,7,None,None,3]]


#woot! I can do these too.
challenger = [[None,None,None,None,4,None,None,None,None], 
              [6,None,1,3,None,8,None,None,None],
              [None,None,4,6,None,None,1,None,8],
              [None,7,None,5,None,None,None,2,None],
              [4,None,None,None,2,None,None,None,6],
              [None,5,None,None,None,6,None,1,None],
              [5,None,7,None,None,2,3,None,None],
              [None,None,None,9,None,3,7,None,4],
              [None,None,None,None,8,None,None,None,None]]
#this board can't be solved by process of elimination. Needs to save state, take a guess, and try
difficult = [[None, 6, None, None, None, None, None, 2, None],
               [None, None, 3, None, None, 7, 1, None, None],
               [None, 1, None, None, 3, 4, None, None, None],
               [6, None, 1, None, None, 2, None, None, None],
               [None, None, None, None, 7, None, None, None, None],
               [None, None, None, 1, None, None, 4, None, 3],
               [None, None, None, 9, 8, None, None, 4, None],
               [None, None, 2, 7, None, None, 9, None, None],
               [None, 5, None, None, None, None,None, 3, None]]
#but it can do this one :)
six = [[None, None, None, None, 6, None],
       [2,3,6,4,1,None],
       [None, 6, None, None, 3, None],
       [None, 2, None, None, 5, None],
       [None, 5,2,6,4,3],
       [None,4,None,None,None,None]]

sixteen =    [[None, 11, 9, None, None, 16, 13, 4, None, None, 14, None, 10, 6, 15, None]    ,
  [4, 12,15, None, 3, 6, None, 11, None, 5, None, 1, 16, 7, 14, 2],
   [1,None,6,None,15,2,None,None,11,9,10,None,None,None,8,None],
   [None,13,None,None,None,1,None,None,4,6,None,15,None,None,None,None],
   [None,None,None,None,None,None,15,None,8,1,5,3,None,4,11,7],
   [6,None,1,None,None,12,8,None,9,None,None,2,None,None,3,None],
   [14,None,4,13,6,None,None,3,None,12,7,10,8,None,2,None],
   [3,8,None,None,4,7,2,None,6,None,None,None,None,12,16,5],
   [13,None,None,16,None,8,14,10,3,4,15,None,12,5,1,11],
   [None,None,None,6,2,None,None,1,10,None,11,None,15,3,None,9],
   [7,None,None,12,None,4,None,15,5,None,9,14,None,None,None,None],
   [10,None,None,8,None,None,11,None,None,None,1,12,4,None,13,16],
   [None,None,None,None,None,None,7,None,15,2,None,None,None,None,12,3],
   [None,None,7,None,None,10,6,None,1,8,None,13,11,None,9,14],
   [8,6,5,None,None,3,None,None,14,None,None,9,None,None,None,None],
   [None,16,None,2,None,None,None,14,None,10,None,None,None,None,None,None]]
print([len(row) for row in sixteen])
board = Sudoku(sixteen)
print(board)
board.solve()
print(board)

class SudokuTester(TestCase) :

     
    def test_solve_hard(self) :
        '''normal sudoku solve'''
        hard_s = Sudoku(hard)
        hard_orig = hard_s.clone()
        hard_solve = hard_s.clone()
        hard_solve.solve()
        self.assert_sudoku_solved(hard_orig, hard_solve)
    def test_solve_six(self) :
        '''test solve a board that's not a perfect square sides'''
        six_orig = Sudoku(six,3,2)
        six_solve= six_orig.clone()
        six_solve.solve()
        self.assert_sudoku_solved(six_orig, six_solve)
    def test_saved_state(self) :
        '''test a really challenging puzzle'''
        difficult_orig = Sudoku(difficult)
        difficult_solve = difficult_orig.clone()
        difficult_solve.solve()
        self.assert_sudoku_solved(difficult_orig, difficult_solve)
    def assert_sudoku_solved(self, initial_board, final_board) :
        #check rows
        initial = initial_board.get_grid()
        final = final_board.get_grid()
        for initial_row, final_row in zip(initial,final) :
            nums = [False] * len(initial_row)
            for initial_cell, final_cell in zip(initial_row, final_row) :
                if initial_cell :
                    self.assertEquals(initial_cell, final_cell)
                self.assertFalse(nums[final_cell - 1])
                nums[final_cell - 1] = True
        
        #check columns
        for j in range(len(initial[0])) :
            initial_col = [initial[i][j] for i in range(len(initial))]
            final_col = [final[i][j] for i in range(len(final))]
            nums = [False] * len(initial_col)
            for initial_cell, final_cell in zip(initial_col, final_col) :
                if initial_cell :
                    self.assertEquals(initial_cell, final_cell)
             
                self.assertFalse(nums[final_cell - 1])
                nums[final_cell - 1] = True
        #check boxes
        #this one needs information from the board in order to properly calculate
        #also, only need to check final here
        for i in range(final_board.col_block) :
            for j in range(final_board.row_block) :
                cells = [x.number for x in final_board.get_block(i,j).cells]
                for k in range(1,len(cells) + 1) :
                    self.assertTrue(k in cells,msg = "%d not found in %d,%d" % (k,i,j)) 
        
          


if __name__ == "__main__" :
    import sys
    if len(sys.argv) - 1 :
        if sys.argv[1] == '-r' :
            s = make_puzzle(sys.stdin)
        elif sys.argv[1] == '-f' and len(sys.argv) > 2 :
            s = make_puzzle(open(sys.argv[2]))
        else :
                print('ERROR: imporoper argument {0}'.format(sys.argv[1]), file=sys.stderr)
                sys.exit(1)
        print(s)
        print('\n')
        s.solve()
        print(s)
    else :
        main()
