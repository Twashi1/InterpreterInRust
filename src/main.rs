use core::panic;
use std::{collections, fmt::LowerExp, thread::current};

// https://rust-unofficial.github.io/patterns/patterns/behavioural/visitor.html

// TODO: could extend number to integer, float, etc.
//      maybe from that figure out traits to more easily perform operations
// TODO: remove pub, i think only needed because was in some namespace before
pub enum Node {
    None,
    Number(f32),
    Boolean(bool),
    BinaryAdd([Box<Node>; 2]),
    BinarySub([Box<Node>; 2]),
    BinaryMul([Box<Node>; 2]),
    BinaryDiv([Box<Node>; 2]),
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
    Compound(Vec<Node>)
}

// TODO: store memory in here
struct InterpreterContext<'mem> {
    memory : &'mem mut collections::BTreeMap<String, Node>
}

// TODO: Store token stream, token index, memory
struct ParserContext<'mem> {
    token_stream : &'mem Vec<Token>,
    token_index: usize
}

// TODO: Store code string, string index
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
        Node::BinaryAdd(children) => return visit_add(&children[0], &children[1], interpreter_context),
        Node::BinarySub(children) => return visit_sub(&children[0], &children[1], interpreter_context),
        Node::BinaryMul(children) => return visit_mul(&children[0], &children[1], interpreter_context),
        Node::BinaryDiv(children) => return visit_div(&children[0], &children[1], interpreter_context),
        Node::Compound(children) => return visit_compound(children, interpreter_context),
        Node::Number(value) => return Node::Number(*value),
        Node::Variable(string) => return visit_variable(string, interpreter_context),
        Node::Assign(string, value) => return visit_assign(string, value, interpreter_context),
        Node::Boolean(value) => return Node::Boolean(*value),
        Node::BooleanEqual(children) => return visit_equal(&children[0], &children[1], interpreter_context),
        Node::BooleanNotEqual(children) => return visit_not_equal(&children[0], &children[1], interpreter_context),
        Node::BooleanLess(children) => return visit_less(&children[0], &children[1], interpreter_context),
        Node::BooleanLessEqual(children) => return visit_less_equal(&children[0], &children[1], interpreter_context),
        Node::BooleanGreater(children) => return visit_more(&children[0], &children[1], interpreter_context),
        Node::BooleanGreaterEqual(children) => return visit_more_equal(&children[0], &children[1], interpreter_context),
        // TODO
        Node::BooleanOr(children) => return Node::None,
        Node::BooleanAnd(children) => return Node::None,
        Node::BooleanNot(node) => return Node::None,
        Node::None => return Node::None
    }
}

fn copy_value(node : &Node) -> Node {
    if let Node::Number(value) = node {
        return Node::Number(*value);
    }

    panic!("Expected variable to have numerical value");
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

fn visit_binary_number(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> [f32; 2] {
    let visited_left : Node = visit(left, interpreter_context);
    let visited_right : Node = visit(right, interpreter_context);

    // Attempt to convert both to number types
    if let Node::Number(left_value) = visited_left {
        if let Node::Number(right_value) = visited_right {
            return [left_value, right_value];
        }
    }

    panic!("One side of binary add was invalid");
}

fn visit_less(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right, interpreter_context);

    return Node::Boolean(values[0] < values[1]);
}

fn visit_more(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right, interpreter_context);

    return Node::Boolean(values[0] > values[1]);
}

fn visit_less_equal(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right, interpreter_context);

    return Node::Boolean(values[0] <= values[1]);
}

fn visit_more_equal(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right, interpreter_context);

    return Node::Boolean(values[0] >= values[1]);
}

fn visit_equal(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    // TODO: could be f32/bool
    let values: [f32; 2] = visit_binary_number(left, right, interpreter_context);

    return Node::Boolean(values[0] == values[1]);
}

fn visit_not_equal(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    // TODO: could be f32/bool
    let values: [f32; 2] = visit_binary_number(left, right, interpreter_context);

    return Node::Boolean(values[0] != values[1]);
}

fn visit_add(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right, interpreter_context);

    return Node::Number(values[0] + values[1]);
}

fn visit_sub(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right, interpreter_context);

    return Node::Number(values[0] - values[1]);
}

fn visit_div(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right, interpreter_context);

    return Node::Number(values[0] / values[1]);
}

fn visit_mul(left: &Node, right: &Node, interpreter_context: &mut InterpreterContext) -> Node {
    let values: [f32; 2] = visit_binary_number(left, right, interpreter_context);

    return Node::Number(values[0] * values[1]);
}

enum Token {
    None,
    Number(f32),
    Boolean(bool),
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
    OpenBrace,
    CloseBrace,
    StatementEnd
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
        _ => panic!("Couldn't copy token")
    }
}

fn read_identifier(string : &String, pos : &mut usize) -> Token {
    // TODO: if/else
    let keywords : collections::BTreeMap<String, Token> = collections::BTreeMap::from([
        (String::from("if"), Token::None),
        (String::from("else"), Token::None),
        (String::from("true"), Token::Boolean(true)),
        (String::from("false"), Token::Boolean(false)),
        (String::from("and"), Token::And),
        (String::from("or"), Token::Or),
        (String::from("not"), Token::Not)
    ]);

    let start: usize = *pos;
    
    while *pos < string.len() {
        let current_char = string.as_bytes()[*pos];

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
        let new_token : Token;

        // Read identifier
        if current_char.is_ascii_alphabetic() {
            new_token = read_identifier(&lexer_context.program, &mut lexer_context.program_index);
        } else {
            match current_char {
                b'0'..=b'9' | b'.' => {
                    new_token = read_number(&lexer_context.program, &mut lexer_context.program_index);
                    lexer_context.program_index -= 1;
                }
                b'+' => new_token = Token::Add,
                b'-' => new_token = Token::Subtract,
                b'*' => new_token = Token::Multiply,
                b'/' => new_token = Token::Divide,
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
                _ => panic!("Couldn't match character: {current_char}")
            }
        }

        tokens.push(new_token);
        lexer_context.program_index += 1;
    }

    return tokens;
}

enum PrecedenceLevels {
    Expression,
    AddSub,
    MulDiv,
    Comparison,
    Equivalence,
    Or,
    And,
    Not,
    Value
}

impl PrecedenceLevels {
    pub fn higher(value : &PrecedenceLevels) -> PrecedenceLevels {
        match value {
            PrecedenceLevels::Expression =>     return PrecedenceLevels::AddSub,
            PrecedenceLevels::AddSub =>         return PrecedenceLevels::MulDiv,
            PrecedenceLevels::MulDiv =>         return PrecedenceLevels::Comparison,
            PrecedenceLevels::Comparison =>     return PrecedenceLevels::Equivalence,
            PrecedenceLevels::Equivalence =>    return PrecedenceLevels::Or,
            PrecedenceLevels::Or =>             return PrecedenceLevels::And,
            PrecedenceLevels::And =>            return PrecedenceLevels::Not,
            PrecedenceLevels::Not =>            return PrecedenceLevels::Value,
            PrecedenceLevels::Value =>          panic!("Can't have higher precedence than value")
        }
    }
}

fn precedence_value(parser_context : &mut ParserContext) -> Node {
    match &parser_context.token_stream[parser_context.token_index] {
        Token::Number(value) => { parser_context.token_index += 1; return Node::Number(*value) },
        Token::Identifier(name) => { parser_context.token_index += 1; return Node::Variable(String::from(name)) }
        _ => panic!("Expected number or identifier")
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
            PrecedenceLevels::AddSub => {
                parser_context.token_index += 1;

                match current_token {
                    Token::Add => lhs = Node::BinaryAdd([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    Token::Subtract => lhs = Node::BinarySub([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    _ => { parser_context.token_index -= 1; return lhs; }
                }
            },
            PrecedenceLevels::MulDiv => {
                parser_context.token_index += 1;

                match current_token {
                    Token::Multiply => lhs = Node::BinaryMul([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
                    Token::Divide => lhs = Node::BinaryDiv([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
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
            PrecedenceLevels::Or => {
                parser_context.token_index += 1;

                match current_token {
                    Token::Or => lhs = Node::BooleanOr([Box::new(lhs), Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))]),
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
            PrecedenceLevels::Not => {
                parser_context.token_index += 1;

                match current_token {
                    Token::And => lhs = Node::BooleanNot(Box::new(precedence_expression(parser_context, PrecedenceLevels::higher(&precedence)))),
                    _ => { parser_context.token_index -= 1; return lhs; }
                }
            },
            PrecedenceLevels::Expression => return lhs,
            _ => panic!("Invalid precedence")
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
            } 
            Token::Number(_) => return precedence_expression(parser_context, PrecedenceLevels::Expression),
            _ => return Node::None
        }
    }

    return Node::None;
}

// Build an AST from a token stream
// Recursive descent parser
fn parse(parser_context : &mut ParserContext) -> Node {
    let mut statements : Vec<Node> = Vec::new();
    
    while parser_context.token_index < parser_context.token_stream.len() {
        // Read a statement
        statements.push(statement(parser_context));

        if parser_context.token_index < parser_context.token_stream.len() {
            let current_token : &Token = &parser_context.token_stream[parser_context.token_index];
        
            // Expect an end token
            if let Token::StatementEnd = current_token {
                parser_context.token_index += 1;
            }
            else { break; }
        }
    }
    
    return Node::Compound(statements);
}

fn interpret(tree : Node, interpreter_context: &mut InterpreterContext) -> Node {
    return visit(&tree, interpreter_context);
}

fn main() {
    let mut lexer_context : LexerContext = LexerContext::new(String::from("5*4+4-3"));
    let tokens : Vec<Token> = lex(&mut lexer_context);

    let mut parser_context : ParserContext = ParserContext::new(&tokens);
    let tree : Node = parse(&mut parser_context);
    
    let mut memory : collections::BTreeMap<String, Node> = collections::BTreeMap::<String, Node>::new();
    let mut interpreter_context : InterpreterContext = InterpreterContext::new(&mut memory);
    let result : Node = interpret(tree, &mut interpreter_context);

    if let Node::Number(value) = result {
        println!("Res: {value}");
    } else {
        panic!("Interpreter didn't return number");
    }
}
