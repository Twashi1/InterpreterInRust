use core::{fmt, panic};
use std::{collections, io::{stdin, Write}};

// https://rust-unofficial.github.io/patterns/patterns/behavioural/visitor.html

// TODO: could extend number to integer, float, etc.
//      maybe from that figure out traits to more easily perform operations
// TODO: remove pub, i think only needed because was in some namespace before

enum Node {
    None,
    Number(f32),
    Boolean(bool),
    String(String),
    InterpolatedString(String),
    BinaryAdd([Box<Node>; 2]),
    BinarySub([Box<Node>; 2]),
    BinaryMul([Box<Node>; 2]),
    BinaryDiv([Box<Node>; 2]),
    UnaryNegation(Box<Node>),
    BooleanGreater([Box<Node>; 2]),
    BooleanGreaterEqual([Box<Node>; 2]),
    BooleanLessEqual([Box<Node>; 2]),
    BooleanLess([Box<Node>; 2]),
    BooleanEqual([Box<Node>; 2]),
    BooleanNotEqual([Box<Node>; 2]),
    BooleanNot(Box<Node>),
    BooleanOr([Box<Node>; 2]),
    BooleanAnd([Box<Node>; 2]),
    Variable(String),
    Assign(String, Box<Node>),
    // Condition, then compound, then else tree
    If(Box<Node>, Box<Node>, Option<Box<Node>>),
    Display(Box<Node>),
    Prompt(Box<Node>),
    Compound(Vec<Node>)
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Node::None => write!(f, "None"),
            Node::Number(value) => write!(f, "Number({value})"),
            Node::Boolean(value) => write!(f, "Boolean({value})"),
            Node::String(value) => write!(f, "String({value})"),
            Node::InterpolatedString(value) => write!(f, "InterpolatedString({value})"),
            Node::BinaryAdd(_) => write!(f, "Add"),
            Node::BinarySub(_) => write!(f, "Subtract"),
            Node::BinaryMul(_) => write!(f, "Multiply"),
            Node::BinaryDiv(_) => write!(f, "Divide"),
            Node::UnaryNegation(_) => write!(f, "Negation"),
            Node::BooleanGreater(_) => write!(f, "Greater"),
            Node::BooleanGreaterEqual(_) => write!(f, "GreaterEqual"),
            Node::BooleanLessEqual(_) => write!(f, "LessEqual"),
            Node::BooleanLess(_) => write!(f, "Less"),
            Node::BooleanEqual(_) => write!(f, "Equal"),
            Node::BooleanNotEqual(_) => write!(f, "NotEqual"),
            Node::BooleanNot(_) => write!(f, "Not"),
            Node::BooleanOr(_) => write!(f, "Or"),
            Node::BooleanAnd(_) => write!(f, "And"),
            Node::Variable(_) => write!(f, "Variable"),
            Node::Assign(_, _) => write!(f, "Assign"),
            Node::If(_, _, _) => write!(f, "If"),
            Node::Display(_) => write!(f, "Display"),
            Node::Prompt(_) => write!(f, "Prompt"),
            Node::Compound(_) => write!(f, "Compound"),
        }
    }
}

struct InterpreterContext<'mem> {
    memory : &'mem mut collections::BTreeMap<String, Node>
}

struct ParserContext<'mem> {
    token_stream : &'mem Vec<Token>,
    token_index: usize
}

struct LexerContext {
    program : String,
    program_index : usize
}

impl LexerContext {
    pub fn new(program : String) -> LexerContext {
        return LexerContext{program: program, program_index: 0};
    }
}

impl ParserContext<'_> {
    pub fn new<'mem>(token_stream : &'mem Vec<Token>) -> ParserContext<'mem> {
        return ParserContext{token_stream: token_stream, token_index: 0 };
    }
}

impl InterpreterContext<'_> {
    pub fn new<'mem>(memory : &'mem mut collections::BTreeMap<String, Node>) -> InterpreterContext<'mem> {
        return InterpreterContext{memory: memory};
    }
}

// TODO: proper "advance" and "peek" functions for lexer/parser/interpreter
// TODO: multi-file project probably soon (or at least namespaces for lexer/parser/interpreter)

// Switch and call correct visitor method
fn visit(n: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    match n {
        Node::BinaryAdd(children)           => return visit_add(&children[0], &children[1], interpreter_context),
        Node::BinarySub(children)           => return visit_sub(&children[0], &children[1], interpreter_context),
        Node::BinaryMul(children)           => return visit_mul(&children[0], &children[1], interpreter_context),
        Node::BinaryDiv(children)           => return visit_div(&children[0], &children[1], interpreter_context),
        Node::UnaryNegation(value)          => return visit_unary_negation(value, interpreter_context),
        Node::Compound(children)            => return visit_compound(children, interpreter_context),
        Node::Number(value)                 => return Node::Number(*value),
        Node::Variable(string)              => return visit_variable(string, interpreter_context),
        Node::Assign(string, value)         => return visit_assign(string, value, interpreter_context),
        Node::Boolean(value)                => return Node::Boolean(*value),
        Node::String(value)                 => return Node::String(String::from(value)),
        Node::InterpolatedString(value)     => return visit_interpolated_string(&value, interpreter_context),
        Node::BooleanEqual(children)        => return visit_equal(&children[0], &children[1], interpreter_context),
        Node::BooleanNotEqual(children)     => return visit_not_equal(&children[0], &children[1], interpreter_context),
        Node::BooleanLess(children)         => return visit_less(&children[0], &children[1], interpreter_context),
        Node::BooleanLessEqual(children)    => return visit_less_equal(&children[0], &children[1], interpreter_context),
        Node::BooleanGreater(children)      => return visit_more(&children[0], &children[1], interpreter_context),
        Node::BooleanGreaterEqual(children) => return visit_more_equal(&children[0], &children[1], interpreter_context),
        Node::BooleanOr(children)           => return visit_or(&children[0], &children[1], interpreter_context),
        Node::BooleanAnd(children)          => return visit_and(&children[0], &children[1], interpreter_context),
        Node::BooleanNot(node)              => return visit_not(&node, interpreter_context),
        Node::If(condition, compound, tree) => return visit_if(&condition, &compound, &tree, interpreter_context),
        Node::Display(node)                 => return visit_display(node, interpreter_context),
        Node::Prompt(node)                  => return visit_prompt(node, interpreter_context),
        Node::None                          => return Node::None
    }
}

type BinaryNumberFunction = fn(f32, f32) -> Node;
type BinaryBooleanFunction = fn(bool, bool) -> Node;
type BinaryFunctionTuple = (Option<BinaryBooleanFunction>, Option<BinaryNumberFunction>);

fn string_convert(node : &Node) -> String {
    match node {
        Node::String(value) => return String::from(value),
        Node::Number(value) => return value.to_string(),
        Node::Boolean(value) => return value.to_string(),
        _ => panic!("No string conversion available for {node}")
    }
}

fn visit_display(node : &Node, interpreter_context : &mut InterpreterContext) -> Node {
    let visited_node : Node = visit(node, interpreter_context);

    let text : String = string_convert(&visited_node);

    print!("{text}");

    return visited_node;
}

fn visit_prompt(node : &Node, interpreter_context : &mut InterpreterContext) -> Node {
    let visited_node : Node = visit(node, interpreter_context);

    let text : String = string_convert(&visited_node);

    print!("{text}");

    match std::io::stdout().flush() {
        Ok(_) => {},
        Err(e) => panic!("Error: {e}")
    }

    let mut input_buffer = String::new();

    match stdin().read_line(&mut input_buffer) {
        Ok(_) => {
            return Node::String(String::from(input_buffer.trim_end()));
        },
        Err(e) => panic!("Error: {e}")
    }
}

fn visit_unary_negation(node : &Node, interpreter_context : &mut InterpreterContext) -> Node {
    let visited_node = visit(node, interpreter_context);
    
    match visited_node {
        Node::Number(value) => return Node::Number(-value),
        _ => panic!("Cannot negate non-number node {node}") 
    }
}

fn visit_interpolated_string(value : &String, interpreter_context : &mut InterpreterContext) -> Node {
    // TODO: extract the formatted parts
    // TODO: build lex and parser for each (seems really time-consuming)
    // TODO: then we lex and parse the formatted parts
    // TODO: then we can interpret the contents using the existing interpreter context
    let mut operating_string : String = String::from(value);
    let mut start_index : usize = usize::MAX;
    let mut current_index : usize = 0;
    let mut is_escaped = false;

    while current_index < operating_string.len() {
        let current_char : u8 = operating_string.as_bytes()[current_index];

        match current_char {
            b'\\' => is_escaped = !is_escaped,
            b'{' => {
                if !is_escaped {
                    // We actually keep the curlies because they're required for a valid program
                    start_index = current_index
                }
            }
            b'}' => {
                if start_index != usize::MAX && !is_escaped {
                    // Keeping the curlies
                    let string_slice : &str = &operating_string[start_index..current_index + 1];
                    
                    let mut lexer_context : LexerContext = LexerContext::new(String::from(string_slice));
                    let token_stream : Vec<Token> = lex(&mut lexer_context);

                    let mut parser_context : ParserContext = ParserContext::new(&token_stream);
                    let tree : Node = parse(&mut parser_context);

                    let result : Node = visit(&tree, interpreter_context);
                    let string_result : String = string_convert(&result);

                    let mut new_string : String = String::from(&operating_string[0..start_index]);
                    new_string.push_str(&string_result);

                    let new_current_index = new_string.len();

                    new_string.push_str(&operating_string[current_index + 1..]);

                    operating_string = new_string;
                    current_index = new_current_index;

                    start_index = usize::MAX;
                }
            }
            _ => is_escaped = false
        }

        current_index += 1;
    }
    
    return Node::String(operating_string);
}

// TODO: string binary operations
fn visit_binary_operation(lhs: &Node, rhs: &Node, functions : BinaryFunctionTuple, interpreter_context : &mut InterpreterContext) -> Node {
    let lhs_visited : Node = visit(lhs, interpreter_context);
    let rhs_visited : Node = visit(rhs, interpreter_context);
    
    // TODO: is there a way to use a jump table here?
    // would have to use numerical value of enum somehow, or associate a numerical value
    // that could be easily extracted from Node and constant

    match lhs_visited {
        Node::Number(lhs_value) => {
            if let Node::Number(rhs_value) = rhs_visited {
                // TODO: 1 is a bit of a magic number
                match functions.1 {
                    Some(function) => return function(lhs_value, rhs_value),
                    None => panic!("No function existed for two number inputs")
                }
            } else {
                panic!("Expected two of same type for binary operation, lhs was {lhs_visited}, rhs was {rhs_visited}");
            }
        },
        Node::Boolean(lhs_value) => {
            if let Node::Boolean(rhs_value) = rhs_visited {
                // TODO: 0 is a bit of a magic number
                match functions.0 {
                    Some(function) => return function(lhs_value, rhs_value),
                    None => panic!("No function existed for two number inputs")
                }
            } else {
                panic!("Expected two of same type for binary operation, lhs was {lhs_visited}, rhs was {rhs_visited}");
            }
        },
        _ => panic!("Expected value type")
    }
}

// TODO: probably prefer another visit call as opposed to copy_value?
fn copy_value(node : &Node) -> Node {
    match node {
        Node::Number(value) => return Node::Number(*value),
        Node::Boolean(value) => return Node::Boolean(*value),
        Node::String(value) => return Node::String(String::from(value)),
        Node::InterpolatedString(value) => return Node::InterpolatedString(String::from(value)),
        _ => panic!("Expected variable to have value")
    }
}

fn visit_if(condition : &Node, compound : &Node, else_tree : &Option<Box<Node>>, interpreter_context : &mut InterpreterContext) -> Node {
    let condition_visited = visit(condition, interpreter_context);

    if let Node::Boolean(value) = condition_visited {
        if value {
            return visit(compound, interpreter_context);
        } else {
            match else_tree {
                Some(tree) => return visit(tree, interpreter_context),
                None => return Node::None
            }
        }
    } else {
        panic!("Expected condition to return boolean value");
    }
}

fn visit_or(left: &Node, right: &Node, interpreter_context : &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            Some(|a : bool, b : bool| -> Node { return Node::Boolean(a || b); }),
            None
        ),
        interpreter_context
    );
}

fn visit_and(left: &Node, right: &Node, interpreter_context : &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            Some(|a : bool, b : bool| -> Node { return Node::Boolean(a && b); }),
            None
        ),
        interpreter_context
    );
}

fn visit_not(value: &Node, interpreter_context : &mut InterpreterContext) -> Node {
    if let Node::Boolean(value_visited) = visit(value, interpreter_context) {
        return Node::Boolean(!value_visited);
    } else {
        panic!("Expected boolean value for not");
    }
}

fn visit_assign(string : &String, value : &Node, interpreter_context : &mut InterpreterContext) -> Node {
    let rhs = visit(value, interpreter_context);
    
    if interpreter_context.memory.contains_key(string) {
        *interpreter_context.memory.get_mut(string).unwrap() = rhs;
    } else {
        interpreter_context.memory.insert(String::from(string), rhs);
    }

    return copy_value(&interpreter_context.memory[string]);
}

fn visit_variable(string : &String, interpreter_context: &mut InterpreterContext) -> Node {
    return copy_value(&interpreter_context.memory[string]);
}

fn visit_compound(children : &Vec<Node>, interpreter_context: &mut InterpreterContext) -> Node {
    let mut last_node : Node = Node::None;
    
    for child in children {
        last_node = visit(child, interpreter_context);
    }

    return last_node;
}

fn visit_less(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            None,
            Some(|a : f32, b : f32| -> Node { return Node::Boolean(a < b); })
        ),
        interpreter_context
    );
}

fn visit_more(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            None,
            Some(|a : f32, b : f32| -> Node { return Node::Boolean(a > b); })
        ),
        interpreter_context
    );
}

fn visit_less_equal(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            None,
            Some(|a : f32, b : f32| -> Node { return Node::Boolean(a <= b); })
        ),
        interpreter_context
    );
}

fn visit_more_equal(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            None,
            Some(|a : f32, b : f32| -> Node { return Node::Boolean(a >= b); })
        ),
        interpreter_context
    );
}

fn visit_equal(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            Some(|a : bool, b : bool| -> Node { return Node::Boolean(a == b); }),
            Some(|a : f32, b : f32| -> Node { return Node::Boolean(a == b); })
        ),
        interpreter_context
    );
}

fn visit_not_equal(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            Some(|a : bool, b : bool| -> Node { return Node::Boolean(a != b); }),
            Some(|a : f32, b : f32| -> Node { return Node::Boolean(a != b); })
        ),
        interpreter_context
    );
}

fn visit_add(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            None,
            Some(|a : f32, b : f32| -> Node { return Node::Number(a + b); })
        ),
        interpreter_context
    );
}

fn visit_sub(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            None,
            Some(|a : f32, b : f32| -> Node { return Node::Number(a - b); })
        ),
        interpreter_context
    );
}

fn visit_div(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            None,
            Some(|a : f32, b : f32| -> Node { return Node::Number(a / b); })
        ),
        interpreter_context
    );
}

fn visit_mul(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit_binary_operation(
        left,
        right,
        (
            None,
            Some(|a : f32, b : f32| -> Node { return Node::Number(a * b); })
        ),
        interpreter_context
    )
}

enum Token {
    None,
    Number(f32),
    Boolean(bool),
    String(String),
    InterpolatedString(String),
    Add,
    Subtract,
    Divide,
    Multiply,
    Assign,
    Identifier(String),
    Equal,
    NotEqual,
    Greater,
    Less,
    GreaterEqual,
    LessEqual,
    Not,
    Or,
    And,
    If,
    Else,
    OpenBrace,
    CloseBrace,
    OpenParen,
    CloseParen,
    Display,
    Prompt,
    StatementEnd
}

impl fmt::Display for Token {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::None => write!(f, "None"),
            Token::Number(value) => write!(f, "Number({value})"),
            Token::Boolean(value) => write!(f, "Bool({value})"),
            Token::String(value) => write!(f, "String({value})"),
            Token::InterpolatedString(value) => write!(f, "IString({value})"),
            Token::Add => write!(f, "Add"),
            Token::Subtract => write!(f, "Subtract"),
            Token::Divide => write!(f, "Divide"),
            Token::Multiply => write!(f, "Multiply"),
            Token::Assign => write!(f, "Assign"),
            Token::Identifier(_) => write!(f, "Identifier"),
            Token::Equal => write!(f, "Equal"),
            Token::NotEqual => write!(f, "NotEqual"),
            Token::Greater => write!(f, "Greater"),
            Token::Less => write!(f, "Less"),
            Token::GreaterEqual => write!(f, "GreaterEqual"),
            Token::LessEqual => write!(f, "LessEqual"),
            Token::Not => write!(f, "Not"),
            Token::Or => write!(f, "Or"),
            Token::And => write!(f, "And"),
            Token::If => write!(f, "If"),
            Token::Else => write!(f, "Else"),
            Token::OpenBrace => write!(f, "OpenBrace"),
            Token::CloseBrace => write!(f, "CloseBrace"),
            Token::OpenParen => write!(f, "OpenParen"),
            Token::CloseParen => write!(f, "CloseParen"),
            Token::Display => write!(f, "Display"),
            Token::Prompt => write!(f, "Prompt"),
            Token::StatementEnd => write!(f, "StatementEnd")
        }
    }
}

fn read_number(string : &String, pos : &mut usize) -> Token {
    let mut current_value : f32 = 0.0;
    let mut digits_after_decimal : i32 = 0;
    
    while *pos < string.len() {
        let current_char : u8 = string.as_bytes()[*pos];

        match current_char {
            b'0'..=b'9' => {
                let digit_value = current_char - b'0';
                
                if digits_after_decimal == 0 {
                    current_value *= 10.0;
                    current_value += digit_value as f32;
                } else {
                    current_value += digit_value as f32 * (10 as f32).powf(-digits_after_decimal as f32) as f32;
                    digits_after_decimal += 1;
                }
            }
            b'.' => {
                if digits_after_decimal > 0 {
                    panic!("Multiple decimal points inside number");
                }

                digits_after_decimal = 1;
            }
            _ => return Token::Number(current_value)
        }

        *pos += 1;
    }

    if current_value != 0.0 {
        return Token::Number(current_value);
    }
    
    panic!("Didn't read integer, or reached EOF");
}

fn copy_token(token : &Token) -> Token {
    match token {
        Token::None => return Token::None,
        Token::Boolean(value) => return Token::Boolean(*value),
        Token::And => return Token::And,
        Token::Or => return Token::Or,
        Token::Not => return Token::Not,
        Token::If => return Token::If,
        Token::Else => return Token::Else,
        Token::Display => return Token::Display,
        Token::Prompt => return Token::Prompt,
        _ => panic!("Couldn't copy token {}", *token)
    }
}

fn read_identifier(string : &String, pos : &mut usize) -> Token {
    let keywords : collections::BTreeMap<String, Token> = collections::BTreeMap::from([
        (String::from("if"), Token::If),
        (String::from("else"), Token::Else),
        (String::from("true"), Token::Boolean(true)),
        (String::from("false"), Token::Boolean(false)),
        (String::from("and"), Token::And),
        (String::from("or"), Token::Or),
        (String::from("not"), Token::Not),
        (String::from("display"), Token::Display),
        (String::from("prompt"), Token::Prompt)
    ]);

    let start: usize = *pos;
    
    while *pos < string.len() {
        let current_char : u8 = string.as_bytes()[*pos];

        if current_char.is_ascii_alphabetic() {
            *pos += 1;
        } else {
            break;
        }
    }

    let identifier: String = String::from(&string[start..*pos]);
    *pos -= 1;
    
    match keywords.get(&identifier) {
        Some(value) => return copy_token(value),
        None => return Token::Identifier(identifier)
    }
}

fn read_string(string : &String, pos : &mut usize) -> Token {
    // Advance past first quotation mark
    *pos += 1;

    let start : usize = *pos;
    let mut is_escaped : bool = false;

    while *pos < string.len() {
        let current_char : u8 = string.as_bytes()[*pos];

        match current_char {
            b'\\' => is_escaped = !is_escaped,
            b'"' => {
                if !is_escaped {
                    break;
                }
            }
            _ => is_escaped = false
        }

        *pos += 1;
    }

    return Token::String(String::from(&string[start..*pos]));
}

fn read_istring(string : &String, pos : &mut usize) -> Token {
    // Advance past first quotation mark
    *pos += 1;

    let start : usize = *pos;
    let mut is_escaped : bool = false;

    while *pos < string.len() {
        let current_char : u8 = string.as_bytes()[*pos];

        match current_char {
            b'\\' => is_escaped = !is_escaped,
            b'`' => {
                if !is_escaped {
                    break;
                }
            }
            _ => is_escaped = false
        }

        *pos += 1;
    }

    return Token::InterpolatedString(String::from(&string[start..*pos]));
}

fn lex_peek(lexer_context : &LexerContext) -> u8 {
    let next_index : usize = lexer_context.program_index + 1;

    if next_index < lexer_context.program.len() {
        return lexer_context.program.as_bytes()[next_index];
    } else {
        return b'\0';
    }
}

fn lex(lexer_context : &mut LexerContext) -> Vec<Token> {
    let mut tokens : Vec<Token> = Vec::new();

    while lexer_context.program_index < lexer_context.program.len() {
        let current_char : u8 = lexer_context.program.as_bytes()[lexer_context.program_index];
        let mut new_token : Token = Token::None;

        // Read identifier
        if current_char.is_ascii_alphabetic() {
            new_token = read_identifier(&lexer_context.program, &mut lexer_context.program_index);
        } else {
            match current_char {
                b' ' | b'\n' | b'\t' => {},
                b'0'..=b'9' | b'.' => {
                    new_token = read_number(&lexer_context.program, &mut lexer_context.program_index);
                    lexer_context.program_index -= 1;
                }
                b'+' => new_token = Token::Add,
                b'-' => new_token = Token::Subtract,
                b'*' => new_token = Token::Multiply,
                b'/' => new_token = Token::Divide,
                b'{' => new_token = Token::OpenBrace,
                b'}' => new_token = Token::CloseBrace,
                b'(' => new_token = Token::OpenParen,
                b')' => new_token = Token::CloseParen,
                b'=' => {
                    match lex_peek(&lexer_context) {
                        b'=' => { new_token = Token::Equal; lexer_context.program_index += 1; },
                        _ => new_token = Token::Assign
                    }
                }
                b'<' => {
                    match lex_peek(&lexer_context) {
                        b'=' => { new_token = Token::LessEqual; lexer_context.program_index += 1; },
                        _ => new_token = Token::Less,
                    }
                }
                b'>' => {
                    match lex_peek(&lexer_context) {
                        b'=' => { new_token = Token::GreaterEqual; lexer_context.program_index += 1; },
                        _ => new_token = Token::Greater,
                    }
                }
                b'!' => {
                    match lex_peek(&lexer_context) {
                        b'=' => { new_token = Token::NotEqual; lexer_context.program_index += 1; },
                        _ => panic!("Unknown token '!{current_char}'")
                    }
                }
                b';' => {
                    new_token = Token::StatementEnd;
                }
                b'"' => {
                    new_token = read_string(&lexer_context.program, &mut lexer_context.program_index);
                }
                b'`' => {
                    new_token = read_istring(&lexer_context.program, &mut lexer_context.program_index);
                }
                _ => panic!("Couldn't match character: {current_char}")
            }
        }

        // Read whitespace
        if let Token::None = new_token {}
        else {
            tokens.push(new_token);
        }

        lexer_context.program_index += 1;
    }

    return tokens;
}

enum PrecedenceLevels {
    Expression,
    Not,
    Or,
    And,
    Equivalence,
    Comparison,
    AddSub,
    MulDiv,
    Value
}

impl PrecedenceLevels {
    pub fn higher(value : &PrecedenceLevels) -> PrecedenceLevels {
        match value {
            PrecedenceLevels::Expression =>     return PrecedenceLevels::Not,
            PrecedenceLevels::Not =>            return PrecedenceLevels::Or,
            PrecedenceLevels::Or =>             return PrecedenceLevels::And,
            PrecedenceLevels::And =>            return PrecedenceLevels::Equivalence,
            PrecedenceLevels::Equivalence =>    return PrecedenceLevels::Comparison,
            PrecedenceLevels::Comparison =>     return PrecedenceLevels::AddSub,
            PrecedenceLevels::AddSub =>         return PrecedenceLevels::MulDiv,
            PrecedenceLevels::MulDiv =>         return PrecedenceLevels::Value,
            PrecedenceLevels::Value =>          panic!("Can't have higher precedence than value")
        }
    }
}

fn precedence_value(parser_context : &mut ParserContext) -> Node {
    let current_token : &Token = &parser_context.token_stream[parser_context.token_index];
    
    parser_context.token_index += 1;

    match current_token {
        Token::Number(value) => return Node::Number(*value),
        Token::Boolean(value) => return Node::Boolean(*value),
        Token::Identifier(name) => return Node::Variable(String::from(name)),
        Token::String(value) => return Node::String(String::from(value)),
        Token::InterpolatedString(value) => return Node::InterpolatedString(String::from(value)),
        Token::Subtract => return Node::UnaryNegation(Box::new(precedence_value(parser_context))),
        Token::Prompt => return Node::Prompt(Box::new(precedence_expression(parser_context, PrecedenceLevels::Expression))),
        Token::Display => return Node::Display(Box::new(precedence_expression(parser_context, PrecedenceLevels::Expression))),
        Token::OpenParen => {
            let inner_expression : Node = precedence_expression(parser_context, PrecedenceLevels::Expression);

            if let Token::CloseParen = parser_context.token_stream[parser_context.token_index] {
                parser_context.token_index += 1;
            } else {
                panic!("Expected closing parenthesis");
            }

            return inner_expression;
        }
        _ => { parser_context.token_index -= 1; panic!("Expected number or identifier") }
    }
}

fn precedence_expression(parser_context : &mut ParserContext, precedence : PrecedenceLevels) -> Node {
    // Base case for recursion
    if let PrecedenceLevels::Value = precedence {
        // Read a value
        return precedence_value(parser_context);
    }

    let mut lhs : Node = precedence_expression(parser_context, PrecedenceLevels::higher(&precedence));

    if parser_context.token_index >= parser_context.token_stream.len() {
        return lhs;
    }

    while parser_context.token_index < parser_context.token_stream.len() {
        let current_token : &Token = &parser_context.token_stream[parser_context.token_index];

        match precedence {
            PrecedenceLevels::Value => panic!("Should never reach value arm of expression parser"),
            PrecedenceLevels::MulDiv => {
                parser_context.token_index += 1;

                match current_token {
                    Token::Multiply => lhs = Node::BinaryMul([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    Token::Divide => lhs = Node::BinaryDiv([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    _ => { parser_context.token_index -= 1; return lhs; }
                }
            },
            PrecedenceLevels::AddSub => {
                parser_context.token_index += 1;

                match current_token {
                    Token::Add => lhs = Node::BinaryAdd([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    Token::Subtract => lhs = Node::BinarySub([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    _ => { parser_context.token_index -= 1; return lhs; }
                }
            },
            PrecedenceLevels::Comparison => {
                parser_context.token_index += 1;

                match current_token {
                    Token::Greater => lhs = Node::BooleanGreater([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    Token::GreaterEqual => lhs = Node::BooleanGreaterEqual([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    Token::Less => lhs = Node::BooleanLess([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    Token::LessEqual => lhs = Node::BooleanLessEqual([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    _ => { parser_context.token_index -= 1; return lhs; }
                }
            },
            PrecedenceLevels::Equivalence => {
                parser_context.token_index += 1;

                match current_token {
                    Token::Equal => lhs = Node::BooleanEqual([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    Token::NotEqual => lhs = Node::BooleanNotEqual([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    _ => { parser_context.token_index -= 1; return lhs; }
                }
            },
            PrecedenceLevels::And => {
                parser_context.token_index += 1;

                match current_token {
                    Token::And => lhs = Node::BooleanAnd([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    _ => { parser_context.token_index -= 1; return lhs; }
                }
            },
            PrecedenceLevels::Or => {
                parser_context.token_index += 1;

                match current_token {
                    Token::Or => lhs = Node::BooleanOr([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    _ => { parser_context.token_index -= 1; return lhs; }
                }
            },
            PrecedenceLevels::Not => {
                parser_context.token_index += 1;

                match current_token {
                    Token::Not => lhs = Node::BooleanNot(Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))),
                    _ => { parser_context.token_index -= 1; return lhs; }
                }
            },
            PrecedenceLevels::Expression => return lhs
        }
    }

    return lhs;
}

const NULL_TOKEN : Token = Token::None;

fn peek<'a>(token_index : &mut usize, token_stream : &'a Vec<Token>) -> &'a Token {
    if *token_index + 1 < token_stream.len() {
        let peek_token : &Token = &token_stream[*token_index + 1];

        return peek_token;
    }
    
    return &NULL_TOKEN;
}

fn parse_compound(parser_context : &mut ParserContext) -> Node {
    // Read curly brace
    if let Token::OpenBrace = parser_context.token_stream[parser_context.token_index] {
        parser_context.token_index += 1;
    } else {
        panic!("Expected open brace at start of compound");
    }

    let mut statements : Vec<Node> = Vec::new();

    // Read statements
    while parser_context.token_index < parser_context.token_stream.len() {
        statements.push(statement(parser_context));

        if parser_context.token_index < parser_context.token_stream.len() {
            let current_token : &Token = &parser_context.token_stream[parser_context.token_index];
        
            parser_context.token_index += 1;

            match current_token {
                Token::StatementEnd => {},
                Token::CloseBrace => return Node::Compound(statements),
                _ => panic!("Expected statement ending or closing brace after statement in compound")
            }
        }
    }

    panic!("Never found closing brace, or was incorrectly eaten");
}

fn parse_if(parser_context : &mut ParserContext) -> Node {
    // Eat the 'if' token
    parser_context.token_index += 1;
    
    // Read the condition
    let condition : Node = precedence_expression(parser_context, PrecedenceLevels::Expression);
    
    // Read the code block
    let compound : Node = parse_compound(parser_context);

    // Read else tree if there is one
    if parser_context.token_index >= parser_context.token_stream.len() {
        return Node::If(Box::new(condition), Box::new(compound), None);
    }

    if let Token::Else = parser_context.token_stream[parser_context.token_index] {
        parser_context.token_index += 1;

        if parser_context.token_index >= parser_context.token_stream.len() {
            panic!("Expected token after else");
        }

        // Chain else if
        if let Token::If = parser_context.token_stream[parser_context.token_index] {
            return Node::If(Box::new(condition), Box::new(compound), Some(Box::new(parse_if(parser_context))));
        }
        // Bare else
        else {
            return Node::If(Box::new(condition), Box::new(compound), Some(Box::new(parse_compound(parser_context))));
        }
    }
    // No else
    else {
        return Node::If(Box::new(condition), Box::new(compound), None);
    }
}

fn statement(parser_context : &mut ParserContext) -> Node {
    while parser_context.token_index < parser_context.token_stream.len() {
        let current_token : &Token = &parser_context.token_stream[parser_context.token_index];

        // TODO: badly formed right now
        match current_token {
            Token::Identifier(name) => {
                match peek(&mut parser_context.token_index, parser_context.token_stream) {
                    Token::Assign => {
                        parser_context.token_index += 2;

                        let rhs: Node = precedence_expression(parser_context, PrecedenceLevels::Expression);

                        return Node::Assign(String::from(name), Box::new(rhs));
                    },
                    _ => {
                        return precedence_expression(parser_context, PrecedenceLevels::Expression);
                    }
                }
            },
            // Any LHS value, including unary negation
            Token::Number(_)                |
            Token::Boolean(_)               |
            Token::String(_)                |
            Token::InterpolatedString(_)    |
            Token::Subtract                 |
            Token::Display                  |
            Token::Prompt                   |
            Token::OpenParen                => return precedence_expression(parser_context, PrecedenceLevels::Expression),
            Token::If => return parse_if(parser_context),
            _ => return Node::None
        }
    }

    return Node::None;
}

// Build an AST from a token stream
// Recursive descent parser
fn parse(parser_context : &mut ParserContext) -> Node {
    // TODO: don't want to require valid programs to be wrapped in curly braces
    return parse_compound(parser_context);
}

fn interpret(tree : Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit(&tree, interpreter_context);
}

fn main() {
    let mut lexer_context : LexerContext = LexerContext::new(String::from("{ x = prompt `Enter your name: `; display `Hi {x}\n` }"));
    let tokens : Vec<Token> = lex(&mut lexer_context);

    let mut i : usize = 0;

    for token in &tokens {
        println!("Token {i}: {token}");

        i += 1;
    }

    let mut parser_context : ParserContext = ParserContext::new(&tokens);
    let tree : Node = parse(&mut parser_context);

    let mut memory : collections::BTreeMap<String, Node> = collections::BTreeMap::<String, Node>::new();
    let mut interpreter_context : InterpreterContext = InterpreterContext::new(&mut memory);
    let result : Node = interpret(tree, &mut interpreter_context);

    match result {
        Node::Number(value) => println!("Result is number {value}"),
        Node::Boolean(value) => println!("Result is bool {value}"),
        Node::String(value) => println!("Result is string {value}"),
        _ => println!("Interpreter didn't return value")
    }
}
