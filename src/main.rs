use core::panic;
use std::collections;

// https://rust-unofficial.github.io/patterns/patterns/behavioural/visitor.html

// TODO: could extend number to integer, float, etc.
//      maybe from that figure out traits to more easily perform operations
// TODO: remove pub, i think only needed because was in some namespace before
pub enum Node {
    None,
    Number(f32),
    BinaryAdd([Box<Node>; 2]),
    BinarySub([Box<Node>; 2]),
    BinaryMul([Box<Node>; 2]),
    BinaryDiv([Box<Node>; 2]),
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
    token_index: usize,
    memory : &'mem mut collections::BTreeMap<String, Node>
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
    pub fn new<'mem>(token_stream : &'mem Vec<Token>, memory: &'mem mut collections::BTreeMap<String, Node> ) -> ParserContext<'mem> {
        return ParserContext{token_stream: token_stream, token_index: 0, memory: memory};
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
    Operator(u8),
    Assign,
    Identifier(String),
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

fn read_identifier(string : &String, pos : &mut usize) -> Token {
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

    return Token::Identifier(identifier);
}

fn lex(lexer_context : &mut LexerContext) -> Vec<Token> {
    let mut tokens : Vec<Token> = Vec::new();

    while lexer_context.program_index < lexer_context.program.len() {
        let current_char : u8 = lexer_context.program.as_bytes()[lexer_context.program_index];

        // Read identifier
        if current_char.is_ascii_alphabetic() {
            tokens.push(read_identifier(&lexer_context.program, &mut lexer_context.program_index));
        } else {
            match current_char {
                b'0'..=b'9' | b'.' => {
                    tokens.push(read_number(&lexer_context.program, &mut lexer_context.program_index));
                    lexer_context.program_index -= 1;
                }
                b'+' | b'-' | b'*' | b'/' => {
                    tokens.push(Token::Operator(current_char));
                }
                b'=' => {
                    tokens.push(Token::Assign);
                }
                b';' => {
                    tokens.push(Token::StatementEnd);
                }
                _ => {}
            }
        }

        lexer_context.program_index += 1;
    }

    return tokens;
}


// TODO: parser shouldn't have to access memory other than to assign value
fn factor(token_index : &mut usize, token_stream : &Vec<Token>, memory : &mut collections::BTreeMap<String, Node>) -> Node {
    if let Token::Number(value) = token_stream[*token_index] {
        *token_index += 1;

        return Node::Number(value);
    }
    else if let Token::Identifier(name) = &token_stream[*token_index] {
        *token_index += 1;

        return Node::Variable(String::from(name));
    }

    panic!("Expected number or identifier");
}

fn term(token_index : &mut usize, token_stream : &Vec<Token>, memory : &mut collections::BTreeMap<String, Node>) -> Node {
    let mut lhs: Node = factor(token_index, token_stream, memory);

    if *token_index >= token_stream.len() {
        return lhs;
    }

    let mut current_token : &Token = &token_stream[*token_index];
    
    while *token_index < token_stream.len() {
        current_token = &token_stream[*token_index];

        if let Token::Operator(operator_type) = current_token {
            match operator_type {
                b'*' => {
                    *token_index += 1;
                    lhs = Node::BinaryMul([Box::new(lhs), Box::new(factor(token_index, token_stream, memory))]);
                },
                b'/' => {
                    *token_index += 1;
                    lhs = Node::BinaryDiv([Box::new(lhs), Box::new(factor(token_index, token_stream, memory))]);
                },
                _ => return lhs
            }
        } else {
            return lhs;
        }
    }

    return lhs;
}

fn expression(token_index : &mut usize, token_stream: &Vec<Token>, memory : &mut collections::BTreeMap<String, Node>) -> Node {
    let mut lhs: Node = term(token_index, token_stream, memory);

    if *token_index >= token_stream.len() {
        return lhs;
    }

    let mut current_token : &Token = &token_stream[*token_index];
    
    while *token_index < token_stream.len() {
        current_token = &token_stream[*token_index];

        if let Token::Operator(operator_type) = current_token {
            match operator_type {
                b'+' => {
                    *token_index += 1;
                    lhs = Node::BinaryAdd([Box::new(lhs), Box::new(term(token_index, token_stream, memory))]);
                },
                b'-' => {
                    *token_index += 1;
                    lhs = Node::BinarySub([Box::new(lhs), Box::new(term(token_index, token_stream, memory))]);
                },
                _ => return lhs
            }
        } else {
            return lhs;
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

fn statement(token_index : &mut usize, token_stream: &Vec<Token>, memory : &mut collections::BTreeMap<String, Node>) -> Node {
    while *token_index < token_stream.len() {
        let current_token : &Token = &token_stream[*token_index];

        // TODO: badly formed right now
        match current_token {
            Token::Identifier(name) => {
                let next_token : &Token = peek(token_index, token_stream);
                
                match next_token {
                    Token::Assign => {
                        *token_index += 2;

                        let rhs: Node = expression(token_index, token_stream, memory);

                        return Node::Assign(String::from(name), Box::new(rhs));
                    },
                    _ => {
                        return expression(token_index, token_stream, memory);
                    }
                }
            } 
            Token::Number(_) => return expression(token_index, token_stream, memory),
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
        statements.push(statement(&mut parser_context.token_index, parser_context.token_stream, &mut parser_context.memory));

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
    let mut lexer_context : LexerContext = LexerContext::new(String::from("x=5+3;x=x+2;x=x/2"));
    let tokens : Vec<Token> = lex(&mut lexer_context);
    let mut memory : collections::BTreeMap<String, Node> = collections::BTreeMap::<String, Node>::new();
    let mut parser_context : ParserContext = ParserContext::new(&tokens, &mut memory);
    let tree : Node = parse(&mut parser_context);
    let mut interpreter_context : InterpreterContext = InterpreterContext::new(&mut memory);
    let result : Node = interpret(tree, &mut interpreter_context);

    if let Node::Number(value) = result {
        println!("Res: {value}");
    } else {
        panic!("Interpreter didn't return number");
    }
}
